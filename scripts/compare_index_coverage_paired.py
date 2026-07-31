"""Does indexing MORE of the corpus actually buy retrieval recall?

The lever. data/bm25_index_general holds 1,600,063 chunks -- every second shard,
because a full build OOM'd -- so the submitted pipeline searches half the corpus it
built. Our answer-support recall@5 is 0.6183 [0.5800,0.6567] against cdeotte's
0.6667 [0.6283,0.7050] on the same rows. Coverage is the obvious suspect, and this
measures whether it actually is one.

Why the slope matters more than any single number. Local RAM caps the build between
1.6M chunks (works) and 2.4M (MemoryError, measured). So a full 3.02M index has to
be built on Kaggle, where a CPU session has ~29-32 GB and costs no GPU quota. That
is worth doing only if coverage moves recall at all. Comparing 1.6M against a 2.0M
build gives the slope for a 25% increase, which is the cheapest evidence available
for whether the remaining 50% is worth the trip.

TWO PASSES, ON PURPOSE. The existing compare_corpus_recall_paired.py loads both
indexes at once; at 1.6M + 2.0M chunks plus both BM25 structures that is the same
OOM that started all this. So each index is scored in a SEPARATE process invocation
and per-row hits are written to disk, then compared. Peak memory is one index, not
two.

Paired bootstrap on per-row answer-support hits, since both indexes score the SAME
rows -- two unpaired CIs would be the wrong test and would likely call a real
effect unresolved (CLAUDE.md).

Usage:
  python scripts/compare_index_coverage_paired.py --score data/bm25_index_general
  python scripts/compare_index_coverage_paired.py --score data/bm25_index_general_full
  python scripts/compare_index_coverage_paired.py --compare
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from llmsci.metrics import bootstrap_ci, paired_bootstrap
from llmsci.retrieve.eval import distinctive_keywords, is_answer_support_hit
from llmsci.retrieve.sparse import BM25Index, build_query

DATA = Path("data")
EVAL_ROWS = 600          # T1 sample; +-0.04 CI half-width on a 0/1 rate at n=600
SEED = 0
K = 5
OUT = Path("reports/index_coverage")
OPTIONS = ["A", "B", "C", "D", "E"]


def eval_rows() -> pd.DataFrame:
    t1 = pd.read_csv(DATA / "t1_dev.csv")
    return t1.sample(n=min(EVAL_ROWS, len(t1)), random_state=SEED).reset_index(drop=True)


def score_index(index_dir: Path) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = eval_rows()
    chunks = pd.read_parquet(index_dir / "chunk_texts.parquet", columns=["text"])
    texts = chunks["text"].tolist()
    del chunks
    print(f"{index_dir.name}: {len(texts):,} chunks indexed", flush=True)
    index = BM25Index.load(index_dir, texts)
    queries = [build_query(r["prompt"], [r[c] for c in OPTIONS]) for _, r in df.iterrows()]
    results = index.search_batch(queries, k=K)
    hits = []
    for (_, row), res in zip(df.iterrows(), results):
        ctx = " ".join(texts[i] for i, _ in res)
        hits.append(1.0 if is_answer_support_hit(ctx, distinctive_keywords(row)) else 0.0)
    arr = np.array(hits)
    m, lo, hi = bootstrap_ci(arr, n_resamples=10_000, seed=0)
    np.save(OUT / f"{index_dir.name}.npy", arr)
    (OUT / f"{index_dir.name}.json").write_text(json.dumps(
        {"index": index_dir.name, "chunks": len(texts), "n_rows": len(df),
         "recall_at_5": [m, lo, hi]}, indent=2))
    print(f"  answer-support recall@{K}: {m:.4f} [{lo:.4f},{hi:.4f}] (n={len(df)})")
    print(f"  wrote {OUT}/{index_dir.name}.npy")


def compare() -> None:
    files = sorted(OUT.glob("*.npy"))
    if len(files) < 2:
        print(f"need two scored indexes in {OUT}/ -- run --score twice first")
        return
    small, big = None, None
    for f in files:
        meta = json.loads((OUT / f"{f.stem}.json").read_text())
        if small is None or meta["chunks"] < small[1]["chunks"]:
            small = (f, meta)
    for f in files:
        meta = json.loads((OUT / f"{f.stem}.json").read_text())
        if f != small[0] and (big is None or meta["chunks"] > big[1]["chunks"]):
            big = (f, meta)
    a, b = np.load(small[0]), np.load(big[0])
    if len(a) != len(b):
        print(f"row counts differ ({len(a)} vs {len(b)}) -- rescore both with the same EVAL_ROWS")
        return
    for tag, meta, arr in ((small[0].stem, small[1], a), (big[0].stem, big[1], b)):
        m, lo, hi = bootstrap_ci(arr, n_resamples=10_000, seed=0)
        print(f"  {tag:28s} {meta['chunks']:>9,} chunks  recall@{K} {m:.4f} [{lo:.4f},{hi:.4f}]")

    d, dlo, dhi = paired_bootstrap(a, b, n_resamples=10_000, seed=0)  # mean(big - small)
    growth = big[1]["chunks"] / small[1]["chunks"]
    resolved = dlo > 0 or dhi < 0
    print(f"\n  paired delta (bigger - smaller): {d:+.4f} [{dlo:+.4f},{dhi:+.4f}]")
    print(f"  corpus grew {growth:.2f}x  |  rows changed: {int((a != b).sum())} of {len(a)}")
    print(f"  {'RESOLVED -- CI excludes 0' if resolved else 'NOT resolved by this eval'}")
    print("\nREADING")
    if resolved and d > 0:
        full = 3_020_431 / big[1]["chunks"]
        print(f"  Coverage IS a real lever (+{d:.4f} for {growth:.2f}x). Extrapolating the")
        print(f"  same slope to the full 3,020,431 chunks ({full:.2f}x more again) is worth a")
        print("  Kaggle CPU session (~29-32 GB RAM, 12 h, zero GPU quota) to build there and")
        print("  attach as a dataset. Extrapolation is NOT a measurement -- verify after.")
    elif not resolved:
        print(f"  Coverage is NOT resolved as a lever at {growth:.2f}x ({d:+.4f}).")
        print("  Do not build the full index on that basis. The 4.8-point deficit against")
        print("  cdeotte's context is then about chunking or ranking, not corpus size, and")
        print("  the honest conclusion is that this pipeline is near what it yields.")
    else:
        print(f"  MORE COVERAGE HURT ({d:+.4f}, CI excludes 0) -- more candidates crowding out")
        print("  good ones at k=5. Would argue for reranking a wider pool, not a bigger index.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", type=Path, help="index dir to score (run once per index)")
    ap.add_argument("--compare", action="store_true")
    args = ap.parse_args()
    if args.score:
        score_index(args.score)
    elif args.compare:
        compare()
    else:
        ap.error("pass --score <index_dir> or --compare")


if __name__ == "__main__":
    main()
