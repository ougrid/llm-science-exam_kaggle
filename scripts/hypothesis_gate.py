"""Cheap, local checks that must pass BEFORE spending GPU hours on a reader run.

Written after four successive explanations for one symptom, three of them wrong
(see DEVLOG.md). Every wrong version had confident arithmetic behind it; what was
missing each time was a measurement that DISCRIMINATES between hypotheses rather
than one that merely agrees with the current favourite.

The trick each check uses: we have a KNOWN-GOOD reader
(`mgoksu/llm-science-run-context-2`, 0.8600 on the clean gold 200 with our
context). A known-good reader is an instrument for testing our DATA. If it
scores well on a dataset, that dataset is learnable and any failure is ours; if
it scores badly, the data is the problem and no amount of reader training will
fix it. That inversion is what none of the earlier debugging did.

Checks, cheapest first:

  1. ALIGNMENT (CPU, seconds). Does each training row's stored context actually
     correspond to that row's query? Re-retrieves a sample and compares. A
     row-shift here would silently destroy training while leaving inference-time
     retrieval (done fresh) perfectly correct -- exactly the failure that would
     look like "the reader can't learn".
  2. CONTEXT QUALITY BY TIER (CPU, seconds). Answer-support recall@5 on the
     training rows vs the eval rows. If our retrieval is much worse on one than
     the other, train/eval are not comparable regardless of source matching.
  3. LEARNABILITY (GPU, ~5 min). The known-good reader scored on OUR eval set
     with OUR context. This is the decisive one: it separates "our context is
     inadequate for this eval set" from "our reader is undertrained". If the
     good reader scores high here, the data is fine and reader training is the
     whole gap -- if it scores near baseline, no reader run can succeed and the
     GPU time should not be spent.

Run: python scripts/hypothesis_gate.py [--skip-gpu]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from llmsci.metrics import bootstrap_ci, random_baseline_map_at_k
from llmsci.retrieve.eval import distinctive_keywords, is_answer_support_hit
from llmsci.retrieve.sparse import BM25Index, build_query

DATA = Path("data")
INDEX_DIR = DATA / "bm25_index_general"
PUBLIC_MODEL = Path("reference_reproduction/models/mgoksu-run-context-2")
OPTIONS = ["A", "B", "C", "D", "E"]
BASELINE = random_baseline_map_at_k()

TRAIN_SRC2 = DATA / "train_pool_own_context_src2.parquet"
T1_CTX = DATA / "t1_dev_own_context_general_big.parquet"

ALIGN_SAMPLE = 25
RECALL_SAMPLE = 300
GPU_SAMPLE = 500
CONTEXT_CHAR_CLIP = 1750


def load_index():
    chunks = pd.read_parquet(INDEX_DIR / "chunk_texts.parquet", columns=["text"])
    texts = chunks["text"].tolist()
    return BM25Index.load(INDEX_DIR, texts), texts


def check_alignment(index, texts) -> bool:
    """Is row i's stored context the context BM25 returns for row i's query?"""
    print("\n[1] ALIGNMENT: does each row's stored context match its own query?")
    ok = True
    for path in (TRAIN_SRC2, T1_CTX):
        if not path.exists():
            print(f"    {path.name}: MISSING -- skipped")
            continue
        df = pd.read_parquet(path)
        sample = df.sample(n=min(ALIGN_SAMPLE, len(df)), random_state=0)
        queries = [build_query(r["prompt"], [r[c] for c in OPTIONS]) for _, r in sample.iterrows()]
        results = index.search_batch(queries, k=5)
        matches = 0
        for (_, row), res in zip(sample.iterrows(), results):
            expected = " ".join(texts[i] for i, _ in res)
            if expected.strip() == str(row["context"]).strip():
                matches += 1
        frac = matches / len(sample)
        verdict = "OK" if frac >= 0.95 else "MISALIGNED"
        if frac < 0.95:
            ok = False
        print(f"    {path.name}: {matches}/{len(sample)} rows reproduce exactly -> {verdict}")
    if not ok:
        print("    => STOP. A row/context shift invalidates training while leaving")
        print("       inference-time retrieval correct, which is indistinguishable")
        print("       from 'the reader cannot learn'.")
    return ok


def check_context_quality(index, texts) -> None:
    """Answer-support recall@5 on training rows vs eval rows."""
    print("\n[2] CONTEXT QUALITY: is retrieval as good on train as on eval?")
    for path in (TRAIN_SRC2, T1_CTX):
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        df = df.sample(n=min(RECALL_SAMPLE, len(df)), random_state=0)
        hits = [
            1.0 if is_answer_support_hit(str(r["context"]), distinctive_keywords(r)) else 0.0
            for _, r in df.iterrows()
        ]
        m, lo, hi = bootstrap_ci(np.array(hits), n_resamples=10_000, seed=0)
        print(f"    {path.name}: answer-support recall@5 {m:.4f} [{lo:.4f},{hi:.4f}] (n={len(df)})")
    print("    => a large train-vs-eval gap here means the two are not comparable,")
    print("       independent of whether their generator sources match.")


def check_learnability() -> None:
    """The decisive check: a KNOWN-GOOD reader on OUR eval set with OUR context."""
    print(f"\n[3] LEARNABILITY: known-good public reader on our eval set + our context")
    if not PUBLIC_MODEL.exists():
        print(f"    {PUBLIC_MODEL} missing -- skipped")
        return
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModelForMultipleChoice, AutoTokenizer

    from llmsci.gpu_guard import cap_memory_fraction
    from llmsci.metrics import average_precision_scores
    from llmsci.reader.mc import DataCollatorForMultipleChoice, logits_to_ranked_labels

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        cap_memory_fraction(0.975)
    df = pd.read_parquet(T1_CTX).sample(n=GPU_SAMPLE, random_state=0).reset_index(drop=True)
    tok = AutoTokenizer.from_pretrained(PUBLIC_MODEL)
    model = AutoModelForMultipleChoice.from_pretrained(PUBLIC_MODEL).to(device).eval()

    class DS(Dataset):
        def __len__(self):
            return len(df)

        def __getitem__(self, i):
            r = df.iloc[i]
            first = f"{str(r['context'])[:CONTEXT_CHAR_CLIP]} #### {r['prompt']}"
            return dict(tok([first] * 5, [str(r[c]) for c in OPTIONS],
                            truncation="only_first", max_length=512))

    loader = DataLoader(DS(), batch_size=2, shuffle=False, collate_fn=DataCollatorForMultipleChoice(tok))
    out = []
    with torch.no_grad():
        for i, b in enumerate(loader):
            b.pop("labels", None)
            out.append(model(**{k: v.to(device) for k, v in b.items()}).logits.float().cpu().numpy())
            if (i + 1) % 50 == 0:
                print(f"      batch {i+1}/{len(loader)}", flush=True)
    logits = np.concatenate(out, axis=0)
    scores = average_precision_scores(df["answer"].tolist(), logits_to_ranked_labels(logits, k=3), k=3)
    m, lo, hi = bootstrap_ci(scores, n_resamples=10_000, seed=0)
    print(f"    known-good reader on T1+our-context: MAP@3 {m:.4f} [{lo:.4f},{hi:.4f}] (n={len(df)})")
    print(f"    our own trained reader on the same data: 0.3840 [0.3649,0.4036]")
    print(f"    the same reader on gold-200+our-context: 0.8600 [0.8208,0.8967]")
    if m > 0.7:
        print("    => VERDICT: our eval data IS learnable with this context. The gap is")
        print("       reader training, and a source-matched run is worth the GPU time.")
    elif m < 0.5:
        print("    => VERDICT: even a known-good reader scores near baseline here, so the")
        print("       problem is our CONTEXT or our EVAL SET, not reader training. A")
        print("       reader run CANNOT fix this -- do not spend the GPU hours.")
    else:
        print("    => VERDICT: partial. Context carries some signal but much less than on")
        print("       the gold 200; expect a source-matched run to land well short of 0.86.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-gpu", action="store_true")
    args = ap.parse_args()
    print(f"random baseline = {BASELINE:.4f}")
    index, texts = load_index()
    print(f"index loaded: {len(texts)} chunks")
    check_alignment(index, texts)
    check_context_quality(index, texts)
    del index, texts
    if not args.skip_gpu:
        check_learnability()


if __name__ == "__main__":
    main()
