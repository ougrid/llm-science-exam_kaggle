"""Modernization: reorder the retrieved context with a 2026 cross-encoder reranker.

The one change from PLAN.md's "2026 stack" applied to this track. Rationale
(measured, see ../NOTES.md): cdeotte's reader has MAX_INPUT=256 and
truncation='only_first', but the retrieved context is a median 1067 tokens and
100% of rows exceed 256. So truncation silently discards ~three quarters of
the retrieved sentences, and because mgoksu concatenated the 20 stage-2
sentences in descending all-MiniLM-L6-v2 cosine order, WHICH sentences reach
the reader is decided entirely by a 2019-era 22M-param bi-encoder's ranking.

Replacing that ordering with `Alibaba-NLP/gte-reranker-modernbert-base`
(149M, Apache-2.0, ModernBERT backbone) is the highest-leverage single swap
available on the data we have: a cross-encoder scores the query and each
sentence jointly instead of comparing independently-pooled vectors. It cannot
add evidence -- it is strictly bounded by mgoksu's stage-1/stage-2 recall --
it can only make better use of the 256-token window.

A stronger embedder was the alternative, but that requires re-running stage 1
over the 13 GB `jjinho/wikipedia-20230701` corpus plus its 9.7 GB FAISS index,
which is out of budget here (and CLAUDE.md forbids a global local dense index).

The query matches mgoksu's stage-2 query exactly: prompt + " " + all five
options concatenated.

Both the eval frame AND the training frames are reranked, then the reader is
retrained on the reranked context. Reranking eval context while training on
MiniLM-ordered context would manufacture exactly the train/inference mismatch
this project spent Day 2 diagnosing.

Usage: python rerank.py
"""

from __future__ import annotations

import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from common import OPTIONS
from llmsci.gpu_guard import cap_memory_fraction

DATA = Path(__file__).resolve().parent.parent / "data"

RERANKER = "Alibaba-NLP/gte-reranker-modernbert-base"
RERANK_MAX_LEN = 512
RERANK_BATCH = 64
# Measured on this box 2026-07-30: 139 ms per 64-pair batch at the synthetic
# probe shape below, peak 1.79 GB. (An earlier 110 ms limit here was written
# before measuring and tripped immediately -- exactly the "don't guess the
# threshold" failure CLAUDE.md warns about; the number below is the measurement.)
# Guard at ~3x to catch the WSL2 shared-memory fallback (15-25x slower) without
# tripping on variance from ragged sentence lengths.
MAX_MS_PER_BATCH = 420

# Sentence-boundary split. The stored context was blingfire-split then joined
# with " ", and blingfire is not installed here (and adding it would mean
# touching the shared venv). This regex splits after ., !, ? or a closing
# quote/paren following one of those, when followed by whitespace and an
# uppercase letter, digit, or a Wikipedia section marker. It does NOT recover
# the original 20 boundaries exactly -- an acknowledged approximation, recorded
# in ../NOTES.md and ../RESULTS.md.
# Two fixed-width lookbehind alternatives -- Python's re rejects the variable
# width form `(?<=[.!?]["')]?)`.
_SPLIT = re.compile(r'(?:(?<=[.!?])|(?<=[.!?]["\')]))\s+(?=[A-Z0-9=*(])')


def split_sentences(context: str, filter_len: int = 3) -> list[str]:
    """Approximate recovery of the retrieved sentence units."""
    if not context or not context.strip():
        return []
    parts = [p.strip() for p in _SPLIT.split(context.strip())]
    return [p for p in parts if len(p) > filter_len]


def build_query(row) -> str:
    """mgoksu's stage-2 query: prompt + " " + A B C D E."""
    answer_all = " ".join(str(row[c]) for c in OPTIONS)
    return f"{row['prompt']} {answer_all}"


@torch.no_grad()
def score_pairs(model, tokenizer, queries: list[str], docs: list[str],
                device, batch_size: int = RERANK_BATCH) -> np.ndarray:
    """Cross-encoder relevance score for each (query, doc) pair."""
    out = []
    for i in range(0, len(queries), batch_size):
        enc = tokenizer(
            queries[i:i + batch_size], docs[i:i + batch_size],
            padding=True, truncation=True, max_length=RERANK_MAX_LEN,
            return_tensors="pt",
        ).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(**enc).logits
        out.append(logits.float().squeeze(-1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


def rerank_frame(df: pd.DataFrame, model, tokenizer, device, label: str,
                 rows_per_chunk: int = 256) -> pd.DataFrame:
    """Add a `context_reranked` column: same sentences, cross-encoder order.

    Pairs are flattened ACROSS rows before hitting the GPU. Scoring one row at
    a time measured 4.6 rows/s (216 ms/row) -- a ~27-pair call underfills the
    GPU and pays per-call tokenization and transfer overhead each time. Batching
    across rows is the same arithmetic, ~4x the throughput.
    """
    df = df.copy()
    reranked: list[str] = [""] * len(df)
    stats: list[int] = [0] * len(df)
    rows = [r._asdict() for r in df.itertuples(index=False)]
    start = time.time()

    for chunk_start in range(0, len(rows), rows_per_chunk):
        chunk = rows[chunk_start:chunk_start + rows_per_chunk]
        queries: list[str] = []
        docs: list[str] = []
        spans: list[tuple[int, int, int, list[str]]] = []  # (row_idx, lo, hi, sents)
        for j, r in enumerate(chunk):
            idx = chunk_start + j
            sents = split_sentences(str(r["context"]))
            stats[idx] = len(sents)
            if len(sents) <= 1:
                reranked[idx] = str(r["context"])
                continue
            q = build_query(r)
            lo = len(docs)
            queries.extend([q] * len(sents))
            docs.extend(sents)
            spans.append((idx, lo, lo + len(sents), sents))

        if docs:
            scores = score_pairs(model, tokenizer, queries, docs, device)
            for idx, lo, hi, sents in spans:
                order = np.argsort(-scores[lo:hi])
                reranked[idx] = " ".join(sents[k] for k in order)

        done = min(chunk_start + rows_per_chunk, len(rows))
        el = time.time() - start
        rate = done / el
        print(f"  [{label}] {done}/{len(rows)} rows  elapsed {el/60:.1f}m "
              f"rate {rate:.1f} rows/s eta {(len(rows)-done)/rate/60:.1f}m", flush=True)

    df["context_reranked"] = reranked
    df["n_sentences"] = stats
    s = np.array(stats)
    print(f"  [{label}] sentences per row: mean {s.mean():.1f} median {np.median(s):.0f} "
          f"p10 {np.percentile(s,10):.0f} p90 {np.percentile(s,90):.0f}")
    return df


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        cap_memory_fraction(0.975)

    tokenizer = AutoTokenizer.from_pretrained(RERANKER)
    model = AutoModelForSequenceClassification.from_pretrained(RERANKER).to(device)
    model.eval()
    print(f"[rerank] {RERANKER} on {device}, "
          f"{sum(p.numel() for p in model.parameters())/1e6:.0f}M params")

    # Speed guard: a real batch at production shape, per CLAUDE.md.
    probe_q = ["what is the mass of an electron " * 5] * RERANK_BATCH
    probe_d = ["The electron mass is about 9.109e-31 kg. " * 5] * RERANK_BATCH
    for _ in range(2):
        score_pairs(model, tokenizer, probe_q, probe_d, device)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(5):
        score_pairs(model, tokenizer, probe_q, probe_d, device)
    torch.cuda.synchronize()
    ms = (time.time() - t0) / 5 * 1000
    print(f"[rerank] {ms:.0f} ms per {RERANK_BATCH}-pair batch (limit {MAX_MS_PER_BATCH})")
    if ms > MAX_MS_PER_BATCH:
        raise RuntimeError(
            f"reranker at {ms:.0f} ms/batch vs limit {MAX_MS_PER_BATCH} -- signature of the "
            "WSL2 shared-GPU-memory fallback or another silent slowdown; see DEVLOG.md"
        )

    for fname, label in [
        ("t1_eval_cdeotte_ctx.parquet", "T1"),
        ("train_8192.parquet", "train"),
        ("monitor_500.parquet", "monitor"),
    ]:
        path = DATA / fname
        df = pd.read_parquet(path)
        print(f"\nreranking {label}: {len(df)} rows")
        out = rerank_frame(df, model, tokenizer, device, label)
        out.to_parquet(path, index=False)
        print(f"  wrote {path}")

    print(f"\npeak GPU {torch.cuda.max_memory_allocated()/1e9:.2f} GB")


if __name__ == "__main__":
    main()
