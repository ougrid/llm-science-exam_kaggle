"""Two half-indexes fused vs one half alone: does full coverage buy recall?

The problem this works around. The general corpus is 3,020,431 chunks but a single
BM25 index over all of it does not fit this box -- measured twice, MemoryError while
tokenising at both 2.4M and 2.0M chunks. The shipped index therefore holds shards
[::2] (1,600,063 chunks) and the submitted pipeline searches HALF the corpus.

The workaround. Index the two halves SEPARATELY -- shards [0::2] and [1::2], each
comfortably buildable -- and fuse their results at query time. Full coverage at half
the peak memory, and no giant index to build or ship. This is also, in effect, what
1st place did: their main retrieval trick was multiple different dumps rather than
one bigger index, and corpus diversity is cheaper than model diversity.

Fusion here is over CHUNKS from disjoint corpora, so there is no shared id space and
no dedupe question -- each index returns its own top-k and we keep the best K
overall. Two ways to decide "best", because they fail differently:

  * SCORE POOL. Take top-K from each, sort the union by raw BM25 score, keep K. BM25
    scores are corpus-dependent (IDF depends on document frequency in THAT index),
    so scores from two different indexes are not strictly comparable -- which is the
    weakness of this method and the reason the next one exists.
  * RRF (k=60). Rank-only, so corpus-dependent score scales cancel. Robust to the
    incomparability above, but throws away the margin information.

Measured as answer-support recall@5 on the same T1 rows, paired bootstrap
throughout, since every arm scores identical rows.

Prints per-arm contribution: how often the fused top-5 draws from the NEW half. If
the answer is that it rarely does, the half we were missing simply does not contain
better evidence, and coverage was never the lever -- which is a real finding, not a
failure, and cheaper to learn here than after a Kaggle build.

Run: python scripts/compare_dual_index_recall.py
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
IDX_A = DATA / "bm25_index_general"           # shards [0::2], what the submission uses
IDX_B = DATA / "bm25_index_general_oddhalf"   # shards [1::2], the missing half
OPTIONS = ["A", "B", "C", "D", "E"]
EVAL_ROWS = 600
SEED = 0
K = 5
RRF_K = 60
CACHE = Path("reports/dual_index")


def retrieve_one(index_dir: Path, queries: list[str], k: int) -> tuple[list[list[str]], list[list[float]]]:
    """Top-k texts and scores per query from ONE index. Loaded and freed in isolation."""
    chunks = pd.read_parquet(index_dir / "chunk_texts.parquet", columns=["text"])
    texts = chunks["text"].tolist()
    del chunks
    index = BM25Index.load(index_dir, texts)
    print(f"  {index_dir.name}: {len(texts):,} chunks", flush=True)
    results = index.search_batch(queries, k=k)
    out_t, out_s = [], []
    for res in results:
        out_t.append([texts[i] for i, _ in res])
        out_s.append([float(s) for _, s in res])
    del index, texts
    return out_t, out_s


def support(texts: list[str], row) -> float:
    return 1.0 if is_answer_support_hit(" ".join(texts), distinctive_keywords(row)) else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=EVAL_ROWS)
    args = ap.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)

    if not IDX_B.exists():
        print(f"{IDX_B} missing -- build it first with:")
        print("  python scripts/build_bm25_index_general_full.py --shard-step 2 "
              "--shard-offset 1 --out-dir data/bm25_index_general_oddhalf")
        return

    t1 = pd.read_csv(DATA / "t1_dev.csv").sample(n=args.rows, random_state=SEED).reset_index(drop=True)
    queries = [build_query(r["prompt"], [r[c] for c in OPTIONS]) for _, r in t1.iterrows()]
    print(f"eval: {len(t1)} T1 rows | answer-support recall@{K}")

    # Sequential, never both indexes resident at once -- that co-residency is the
    # same OOM that forced the half-index in the first place.
    a_texts, a_scores = retrieve_one(IDX_A, queries, K)
    b_texts, b_scores = retrieve_one(IDX_B, queries, K)

    hits_a = np.array([support(a_texts[i], t1.iloc[i]) for i in range(len(t1))])
    hits_b = np.array([support(b_texts[i], t1.iloc[i]) for i in range(len(t1))])

    # SCORE POOL: union both top-K, sort by raw score, keep K.
    pooled, from_b_pool = [], []
    for i in range(len(t1)):
        merged = [(s, txt, "a") for s, txt in zip(a_scores[i], a_texts[i])] + \
                 [(s, txt, "b") for s, txt in zip(b_scores[i], b_texts[i])]
        merged.sort(key=lambda x: -x[0])
        top = merged[:K]
        pooled.append([txt for _, txt, _ in top])
        from_b_pool.append(sum(1 for _, _, src in top if src == "b") / K)
    hits_pool = np.array([support(pooled[i], t1.iloc[i]) for i in range(len(t1))])

    # RRF: rank-only, so the two corpora's incomparable IDF scales cancel.
    rrf_sel, from_b_rrf = [], []
    for i in range(len(t1)):
        scored = {}
        for rank, txt in enumerate(a_texts[i]):
            scored[txt] = scored.get(txt, 0.0) + 1.0 / (RRF_K + rank + 1)
        for rank, txt in enumerate(b_texts[i]):
            scored[txt] = scored.get(txt, 0.0) + 1.0 / (RRF_K + rank + 1)
        top = sorted(scored.items(), key=lambda kv: -kv[1])[:K]
        sel = [txt for txt, _ in top]
        rrf_sel.append(sel)
        from_b_rrf.append(sum(1 for txt in sel if txt in set(b_texts[i])) / K)
    hits_rrf = np.array([support(rrf_sel[i], t1.iloc[i]) for i in range(len(t1))])

    arms = {
        "A only (shipped, shards [0::2])": hits_a,
        "B only (missing half, [1::2])": hits_b,
        "A+B score pool": hits_pool,
        "A+B RRF(k=60)": hits_rrf,
    }
    print()
    for tag, h in arms.items():
        m, lo, hi = bootstrap_ci(h, n_resamples=10_000, seed=0)
        print(f"  {tag:34s} recall@{K} {m:.4f} [{lo:.4f},{hi:.4f}]")
    print(f"\n  fused top-{K} drawn from the NEW half: "
          f"score-pool {np.mean(from_b_pool):.3f}, RRF {np.mean(from_b_rrf):.3f}")

    print(f"\n  PAIRED vs A only (the shipped index), same rows:")
    winners = []
    for tag, h in arms.items():
        if tag.startswith("A only"):
            continue
        d, lo, hi = paired_bootstrap(hits_a, h, n_resamples=10_000, seed=0)  # mean(arm - A)
        mark = "RESOLVED" if (lo > 0 or hi < 0) else "not resolved"
        print(f"    {tag:34s} {d:+.4f} [{lo:+.4f},{hi:+.4f}]  {mark}")
        if d > 0 and lo > 0:
            winners.append((tag, d))

    print()
    if winners:
        tag, d = max(winners, key=lambda x: x[1])
        print(f"COVERAGE IS A REAL LEVER: {tag} beats the shipped index by {d:+.4f}.")
        print("  Ship it: attach BOTH index halves to the submission notebook and fuse at")
        print("  query time. Retrieval cost roughly doubles but retrieval was ~minutes of")
        print("  the 9 h budget, and no reader changes are needed.")
        print("  Then re-measure MAP@3 on the gold 200 -- recall gains do not convert 1:1.")
    else:
        print("NOT RESOLVED. The missing half does not add retrievable answer support for")
        print("  these questions, so corpus coverage was not the lever. That makes the")
        print("  4.8-point deficit against cdeotte's context a chunking or ranking issue,")
        print("  and it means the shipped 0.761131 is close to what this pipeline yields.")
    (CACHE / "dual_index_recall.json").write_text(json.dumps(
        {tag: list(bootstrap_ci(h, n_resamples=10_000, seed=0)) for tag, h in arms.items()}
        | {"frac_from_new_half_pool": float(np.mean(from_b_pool)),
           "frac_from_new_half_rrf": float(np.mean(from_b_rrf)), "n_rows": len(t1)}, indent=2))


if __name__ == "__main__":
    main()
