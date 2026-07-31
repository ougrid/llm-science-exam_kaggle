"""Offline submission: our own retrieval + a PUBLIC reader checkpoint.

PROVENANCE -- READ BEFORE QUOTING ANY SCORE FROM THIS NOTEBOOK.
This submission is a hybrid, and the split matters:
  * OURS: the corpus (jjinho/wikipedia-20230701, sampled to 1.6M chunks),
    the chunking + title-prefixing, the BM25 index, the query construction,
    and this offline inference pipeline.
  * NOT OURS: the reader. `mgoksu/llm-science-run-context-2` is a public
    checkpoint fine-tuned by another competitor -- one leg of the notebook
    behind the published 0.823761. Using public models/datasets is explicitly
    permitted by this competition and was standard practice in it, so this is
    a legitimate submission. It is NOT evidence of a model we trained.
    Cited in CREDITS.md.
Locally, this exact pairing (our context -> this public reader) scored
MAP@3 0.8592 [0.8200, 0.8958] on the clean official 200 (`train.csv`), which is
a lower bound since the reader reads context from a retriever it was never
trained against.

Competition constraints honoured: internet disabled (bm25s installed from an
attached wheel dataset), all models/data attached as Kaggle Datasets, and a
global time budget with a safe fallback so a partial run still emits a valid
full-length submission rather than failing.

Handles the commit/submit asymmetry: ~200 rows are visible at commit time and
~4,000 at submit time. Nothing is short-circuited (the pipeline is fast enough
either way), but row counts are printed so the two phases are distinguishable
in the logs.
"""

from __future__ import annotations

import gc
import os
import subprocess
import sys
import time
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

START = time.time()
TIME_BUDGET_S = 7.5 * 3600  # competition limit is 9 h; leave headroom
OUT = Path("/kaggle/working")
OPTIONS = ["A", "B", "C", "D", "E"]
TOP_K = 5
RRF_K = 60  # reciprocal-rank-fusion constant; matches src/llmsci/retrieve/fuse.py
CONTEXT_CHAR_CLIP = 1750  # matches the reader's own part-2 inference recipe
BATCH_SIZE = 2  # v2 OOM'd on a T4 at 8; 200-4,000 rows is fast either way


def find_path(filename: str) -> Path:
    matches = glob(f"/kaggle/input/**/{filename}", recursive=True)
    if not matches:
        print("DEBUG /kaggle/input tree:")
        for p in sorted(glob("/kaggle/input/**/*", recursive=True))[:200]:
            print(" ", p)
        raise FileNotFoundError(filename)
    return Path(matches[0])


def find_dir(marker: str) -> Path:
    """Directory containing `marker` (e.g. a config.json for the model)."""
    return find_path(marker).parent


def install_bm25s() -> None:
    wheel = glob("/kaggle/input/**/bm25s-*.whl", recursive=True)
    if not wheel:
        raise FileNotFoundError("bm25s wheel not found in attached datasets")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", "-q", wheel[0]]
    )
    print(f"installed {os.path.basename(wheel[0])} offline")


def write_submission(df: pd.DataFrame, preds: list[str]) -> None:
    sub = pd.DataFrame({"id": df["id"].values, "prediction": preds})
    assert len(sub) == len(df), (len(sub), len(df))
    assert sub["prediction"].map(lambda s: 1 <= len(s.split()) <= 3).all()
    assert sub["prediction"].map(lambda s: all(t in OPTIONS for t in s.split())).all()
    assert sub["prediction"].map(lambda s: len(set(s.split())) == len(s.split())).all()
    sub.to_csv(OUT / "submission.csv", index=False)
    print(f"wrote submission.csv with {len(sub)} rows")
    print(sub.head(3).to_string(index=False))


def main() -> None:
    install_bm25s()
    import bm25s  # noqa: E402  (installed above)
    import torch  # noqa: E402
    from torch.utils.data import DataLoader, Dataset  # noqa: E402
    from transformers import AutoModelForMultipleChoice, AutoTokenizer  # noqa: E402

    test = pd.read_csv(find_path("test.csv"))
    print(f"test rows: {len(test)}  ({'COMMIT phase' if len(test) <= 400 else 'SUBMIT phase'})")

    # Fallback written immediately: if anything below fails or runs long, a
    # valid full-length submission already exists on disk. "A B C" scores the
    # analytic random baseline (0.3667) rather than erroring out.
    write_submission(test, ["A B C"] * len(test))

    # DUAL-INDEX RETRIEVAL WITH RRF FUSION.
    # The general corpus is 3,020,431 chunks, but a single BM25 index over all of
    # it will not build in 15 GB -- MemoryError while tokenising at both 2.4M and
    # 2.0M chunks, measured. The old build's SHARD_STEP=2 workaround meant this
    # submission searched HALF the corpus. Fix: index the two halves separately
    # (shards [0::2] and [1::2]) and fuse their results per query.
    # Measured on all 1,500 T1 rows: answer-support recall@5 goes 0.6207 -> 0.6400,
    # a paired +0.0193 [+0.0053,+0.0327] -- RESOLVED. On the clean gold 200 the
    # reader's MAP@3 goes 0.8600 -> 0.8733, +0.0133 [-0.0108,+0.0375], which 200
    # rows cannot resolve; the upstream recall gain is what justifies this.
    # Either half alone scores 0.6207 vs 0.6220 -- identical -- so Wikipedia is
    # redundant and the win comes from fusing two views, not from new topics.
    index_dirs = sorted({Path(m).parent for m in
                         glob("/kaggle/input/**/params.index.json", recursive=True)})
    if not index_dirs:
        raise FileNotFoundError("no BM25 index found under /kaggle/input")
    print(f"BM25 indexes ({len(index_dirs)}): {[d.name for d in index_dirs]}")

    # Identical to src/llmsci/retrieve/sparse.py build_query(), so the submission
    # retrieves exactly what the local gold-200 measurements retrieved.
    queries = [
        f"{r['prompt']} " + " ".join(str(r[c]) for c in OPTIONS) for _, r in test.iterrows()
    ]
    tokenized = bm25s.tokenize(queries, stopwords="en", show_progress=False)

    # One index resident at a time. Holding both was the original OOM, and Kaggle's
    # RAM is generous but not unlimited -- this costs a little wall clock and
    # removes the failure mode entirely.
    per_index_hits: list[list[list[str]]] = []
    for d in index_dirs:
        chunks = pd.read_parquet(d / "chunk_texts.parquet", columns=["text"])
        chunk_texts = chunks["text"].tolist()
        del chunks
        retriever = bm25s.BM25.load(str(d))
        idx, _scores = retriever.retrieve(tokenized, k=TOP_K, show_progress=False)
        per_index_hits.append([[chunk_texts[j] for j in row] for row in idx])
        print(f"  {d.name}: {len(chunk_texts)} chunks, retrieved top-{TOP_K} "
              f"[{time.time() - START:.0f}s]", flush=True)
        del chunk_texts, retriever, idx
        gc.collect()

    # RRF over TEXTS, not chunk ids: the indexes cover disjoint shards so their id
    # spaces are unrelated. Rank-based fusion also sidesteps the fact that BM25
    # scores are not comparable across corpora (IDF depends on each index's own
    # document frequencies) -- which is why RRF beat raw score-pooling locally,
    # +0.0193 vs +0.0167.
    contexts = []
    for row_i in range(len(test)):
        scored: dict[str, float] = {}
        for hits in per_index_hits:
            for rank, txt in enumerate(hits[row_i]):
                scored[txt] = scored.get(txt, 0.0) + 1.0 / (RRF_K + rank + 1)
        top = sorted(scored.items(), key=lambda kv: -kv[1])[:TOP_K]
        contexts.append(" ".join(txt for txt, _ in top))
    test = test.copy()
    test["context"] = contexts
    print(f"fused top-{TOP_K} across {len(index_dirs)} index(es) for {len(test)} rows "
          f"[{time.time() - START:.0f}s]")
    del per_index_hits, tokenized
    gc.collect()

    model_dir = find_dir("config.json")
    print(f"reader: {model_dir}")
    # PLAN.md's env gate, applied here: torch.cuda.is_available() can return True
    # on a GPU whose compute capability this torch build has no kernels for
    # (Kaggle assigned a P100 on the previous attempt -> "no kernel image is
    # available for execution on the device", failing only at first launch).
    # Probe with a real kernel launch and fall back to CPU rather than dying:
    # 200 rows (commit) or ~4,000 (submit) on CPU is slow but finishes inside
    # the 9 h limit, and a slow valid submission beats a fast failed one.
    device = torch.device("cpu")
    if torch.cuda.is_available():
        try:
            _p = torch.zeros(8, 8, device="cuda")
            _ = (_p @ _p).sum().item()
            torch.cuda.synchronize()
            device = torch.device("cuda")
            print(f"GPU usable: {torch.cuda.get_device_name(0)} "
                  f"cc={torch.cuda.get_device_capability(0)}")
        except Exception as e:
            print(f"GPU present but UNUSABLE ({type(e).__name__}: {e}) -- falling back to CPU")
    print(f"device: {device}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForMultipleChoice.from_pretrained(model_dir).to(device).eval()

    class InferDS(Dataset):
        def __init__(self, df):
            self.df = df.reset_index(drop=True)

        def __len__(self):
            return len(self.df)

        def __getitem__(self, i):
            r = self.df.iloc[i]
            first = f"{str(r['context'])[:CONTEXT_CHAR_CLIP]} #### {r['prompt']}"
            return dict(tokenizer([first] * 5, [str(r[c]) for c in OPTIONS],
                                  truncation="only_first", max_length=512))

    def collate(feats):
        n = len(feats)
        flat = [{k: v[i] for k, v in f.items()} for f in feats for i in range(5)]
        b = tokenizer.pad(flat, padding=True, return_tensors="pt")
        return {k: v.view(n, 5, -1) for k, v in b.items()}

    loader = DataLoader(InferDS(test), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)
    all_logits, n_done = [], 0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            try:
                b = {k: v.to(device) for k, v in batch.items()}
                all_logits.append(model(**b).logits.float().cpu().numpy())
            except torch.OutOfMemoryError:
                # Degrade rather than die: finish this batch on CPU.
                torch.cuda.empty_cache()
                model_cpu = model.to("cpu")
                all_logits.append(model_cpu(**batch).logits.float().numpy())
                model.to(device)
                print(f"  batch {i + 1}: OOM -> computed on CPU", flush=True)
            n_done += batch["input_ids"].shape[0]
            if (i + 1) % 25 == 0 or (i + 1) == len(loader):
                el = time.time() - START
                print(f"  batch {i + 1}/{len(loader)} rows {n_done}/{len(test)} [{el:.0f}s]", flush=True)
            if time.time() - START > TIME_BUDGET_S:
                print("TIME BUDGET hit -- keeping the baseline fallback for unprocessed rows")
                break

    logits = np.concatenate(all_logits, axis=0)
    order = np.argsort(-logits, axis=1)[:, :3]
    preds = [" ".join(OPTIONS[j] for j in row) for row in order]
    if len(preds) < len(test):  # time-budget break: pad with the fallback
        preds = preds + ["A B C"] * (len(test) - len(preds))
    write_submission(test, preds)
    print(f"done [{time.time() - START:.0f}s]")


if __name__ == "__main__":
    main()
