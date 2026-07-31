"""SCORE LEVER: how much of the retrieved context should the submission show?

The submitted pipeline (notebooks/kaggle/day3-submission, LB 0.761131) clips
context to 1,750 chars and tokenises at max_length=512. Retrieval returns 5 chunks
averaging ~5,900 chars, so roughly 70% of the evidence is discarded before the
reader ever sees it. Answer-support recall@5 of 0.62 is measured over the FULL five
chunks -- the reader only reads the first third of what that number describes.

Why this is worth testing rather than assuming. `microsoft/deberta-v3-large`'s
config sets max_position_embeddings=512 but ALSO position_biased_input=false and
relative_attention=true with position_buckets=256, so it never adds absolute
position embeddings -- position enters only through log-bucketed relative
attention. The 512 is effectively vestigial and max_length can be raised with no
model surgery. 4th place in this competition reported score rising monotonically
from 512 to 1280 tokens. This project has never tested it.

Nothing here trains anything. It is inference only, on the reader already used by
the submitted notebook, so a win transfers directly to the leaderboard by changing
two constants.

Measured on the clean official 200 (`data/holdout_gold.csv` + our retrieval), which
is the tier the 0.8592 local figure came from, so results are comparable to it.
This spends gold-set evaluations -- CLAUDE.md caps them at ~8 for the project, and
this run uses several at once deliberately, because a paired sweep on the tier that
actually predicts the leaderboard is worth more than the same arms on T1.

Paired bootstrap against the current setting, since every arm scores the SAME 200
rows: two configs differing only in context budget agree on most rows, so the SD of
the per-row difference is far smaller than either arm's own SD.

Run: python scripts/sweep_context_length_gold.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from llmsci.gpu_guard import cap_memory_fraction
from llmsci.metrics import (
    average_precision_scores,
    bootstrap_ci,
    paired_bootstrap,
    random_baseline_map_at_k,
)
from llmsci.reader.mc import DataCollatorForMultipleChoice, logits_to_ranked_labels

DATA = Path("data")
GOLD_CTX = DATA / "gold_own_context_general.parquet"   # built below if absent
GOLD = DATA / "holdout_gold.csv"
INDEX_DIR = DATA / "bm25_index_general"
PUBLIC = Path("reference_reproduction/models/mgoksu-run-context-2")
OPTIONS = ["A", "B", "C", "D", "E"]
BASELINE = random_baseline_map_at_k()
TOP_K = 5

# (chars_of_context, max_length). First row is the SUBMITTED setting -- the
# baseline every other arm is paired against.
ARMS = [
    (1750, 512),    # what the 0.761131 submission does
    (2750, 768),
    (4000, 1024),
    (5500, 1280),   # 4th place's endpoint
]
BATCH = {512: 2, 768: 2, 1024: 1, 1280: 1}  # 8 GB card; longer seqs need smaller batches


def build_gold_context() -> pd.DataFrame:
    """Retrieve for the gold 200 with the same index/query as the submission."""
    if GOLD_CTX.exists():
        return pd.read_parquet(GOLD_CTX)
    from llmsci.retrieve.sparse import BM25Index, build_query

    print("building gold-200 context (one-off)...", flush=True)
    chunks = pd.read_parquet(INDEX_DIR / "chunk_texts.parquet", columns=["text"])
    texts = chunks["text"].tolist()
    index = BM25Index.load(INDEX_DIR, texts)
    df = pd.read_csv(GOLD)
    queries = [build_query(r["prompt"], [r[c] for c in OPTIONS]) for _, r in df.iterrows()]
    results = index.search_batch(queries, k=TOP_K)
    df["context"] = [" ".join(texts[i] for i, _ in res) for res in results]
    df.to_parquet(GOLD_CTX, index=False)
    del index, texts, chunks
    print(f"wrote {GOLD_CTX} ({len(df)} rows)", flush=True)
    return df


def score_arm(model, tok, df, chars: int, maxlen: int, device) -> np.ndarray:
    class DS(Dataset):
        def __len__(self):
            return len(df)

        def __getitem__(self, i):
            r = df.iloc[i]
            first = f"{str(r['context'])[:chars]} #### {r['prompt']}"
            return dict(tok([first] * 5, [str(r[c]) for c in OPTIONS],
                            truncation="only_first", max_length=maxlen))

    loader = DataLoader(DS(), batch_size=BATCH.get(maxlen, 1), shuffle=False,
                        collate_fn=DataCollatorForMultipleChoice(tok))
    out = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16,
                                         enabled=(device.type == "cuda")):
        for b in loader:
            b.pop("labels", None)
            out.append(model(**{k: v.to(device) for k, v in b.items()}).logits.float().cpu().numpy())
    logits = np.concatenate(out, axis=0)
    return average_precision_scores(df["answer"].tolist(), logits_to_ranked_labels(logits, k=3), k=3)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(PUBLIC))
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        cap_memory_fraction(0.975)

    df = build_gold_context()
    ctx_len = df["context"].str.len()
    print(f"gold 200 with our retrieval | context chars: median {ctx_len.median():.0f}, "
          f"p90 {ctx_len.quantile(0.9):.0f} | baseline {BASELINE:.4f}")

    from transformers import AutoModelForMultipleChoice, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForMultipleChoice.from_pretrained(args.model).to(device).eval()

    results = {}
    for chars, maxlen in ARMS:
        try:
            s = score_arm(model, tok, df, chars, maxlen, device)
        except Exception as e:  # WSL2 reports over-budget as "device not ready", not OOM
            print(f"  {chars:>5}ch @ {maxlen:<5} FAILED ({type(e).__name__}: {str(e)[:60]})",
                  flush=True)
            if device.type == "cuda":
                torch.cuda.empty_cache()
            continue
        m, lo, hi = bootstrap_ci(s, n_resamples=10_000, seed=0)
        tag = "  <- SUBMITTED" if (chars, maxlen) == ARMS[0] else ""
        print(f"  {chars:>5}ch @ {maxlen:<5} MAP@3 {m:.4f} [{lo:.4f},{hi:.4f}]{tag}", flush=True)
        results[(chars, maxlen)] = s

    if ARMS[0] not in results:
        print("baseline arm failed -- cannot pair")
        return
    base = results[ARMS[0]]
    print(f"\nPAIRED vs the submitted {ARMS[0][0]}ch @ {ARMS[0][1]} (same 200 rows):")
    best, best_d = None, 0.0
    for key, s in results.items():
        if key == ARMS[0]:
            continue
        d, lo, hi = paired_bootstrap(base, s, n_resamples=10_000, seed=0)  # mean(arm - base)
        mark = "RESOLVED" if (lo > 0 or hi < 0) else "not resolved by 200 rows"
        print(f"  {key[0]:>5}ch @ {key[1]:<5} delta {d:+.4f} [{lo:+.4f},{hi:+.4f}]  {mark}")
        if d > best_d and lo > 0:
            best, best_d = key, d

    print()
    if best:
        print(f"SHIP IT: {best[0]} chars @ max_length={best[1]}  (+{best_d:.4f}, CI excludes 0).")
        print(f"  Change CONTEXT_CHAR_CLIP={best[0]} and max_length={best[1]} in")
        print("  notebooks/kaggle/day3-submission/script.py. Watch the 9 h limit: longer")
        print("  sequences cost roughly linearly, and ~4,000 rows must still finish.")
    else:
        print("No arm beats the submitted setting with a CI excluding 0 on 200 rows.")
        print("  200 rows resolves ~0.04 at best, so a real +0.02 would hide here --")
        print("  report as unresolved rather than shipping on an unresolved delta.")
    Path("reports").mkdir(exist_ok=True)
    Path("reports/context_length_gold.json").write_text(json.dumps(
        {f"{c}ch_{m}": list(bootstrap_ci(s, n_resamples=10_000, seed=0))
         for (c, m), s in results.items()}, indent=2))


if __name__ == "__main__":
    main()
