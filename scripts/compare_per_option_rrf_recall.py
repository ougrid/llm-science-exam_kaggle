"""Does per-option retrieval + RRF beat the single pooled query?

PLAN.md calls this "the single highest-expected-value trick on the list".
Tested here as a pure retrieval metric (no reader, no GPU), paired on the same
T1 rows, so the answer does not depend on the reader -- which matters
especially now that the last-day diagnostic showed the reader in every
previous run was untrained, invalidating every reader-level verdict in
reports/ablation_table.md.

Run from the repo root, after scripts/build_bm25_index_general.py:
`python scripts/compare_per_option_rrf_recall.py`
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from llmsci.experiment import git_sha, log_experiment
from llmsci.metrics import paired_bootstrap
from llmsci.retrieve.eval import distinctive_keywords, is_answer_support_hit
from llmsci.retrieve.fuse import search_per_option_rrf
from llmsci.retrieve.sparse import BM25Index, build_query

DATA = Path("data")
INDEX_DIR = DATA / "bm25_index_general"
OPTION_COLUMNS = ["A", "B", "C", "D", "E"]
K_FINAL = 5
K_PER_QUERY = 20


def hits_from_results(
    df: pd.DataFrame, results: list[list[tuple[int, float]]], chunk_texts: list[str]
) -> np.ndarray:
    hits = []
    for (_, row), candidates in zip(df.iterrows(), results):
        kws = distinctive_keywords(row)
        hit = any(is_answer_support_hit(chunk_texts[idx], kws) for idx, _ in candidates[:K_FINAL])
        hits.append(1.0 if hit else 0.0)
    return np.array(hits)


def main() -> None:
    chunks = pd.read_parquet(INDEX_DIR / "chunk_texts.parquet")
    chunk_texts = chunks["text"].tolist()
    index = BM25Index.load(INDEX_DIR, chunk_texts)
    t1 = pd.read_csv(DATA / "t1_dev.csv")

    prompts = t1["prompt"].tolist()
    options_per_row = [[row[c] for c in OPTION_COLUMNS] for _, row in t1.iterrows()]

    start = time.time()
    pooled_queries = [build_query(p, o) for p, o in zip(prompts, options_per_row)]
    pooled_results = index.search_batch(pooled_queries, k=K_FINAL)
    pooled_seconds = time.time() - start
    pooled_hits = hits_from_results(t1, pooled_results, chunk_texts)

    start = time.time()
    rrf_results = search_per_option_rrf(
        index, prompts, options_per_row, k_per_query=K_PER_QUERY, k_final=K_FINAL
    )
    rrf_seconds = time.time() - start
    rrf_hits = hits_from_results(t1, rrf_results, chunk_texts)

    print(f"pooled  query  recall@{K_FINAL}: {pooled_hits.mean():.4f}  ({pooled_seconds:.1f}s)")
    print(f"per-option RRF recall@{K_FINAL}: {rrf_hits.mean():.4f}  ({rrf_seconds:.1f}s, "
          f"{rrf_seconds / max(pooled_seconds, 1e-9):.1f}x the retrieval cost)")

    diff_mean, diff_lower, diff_upper = paired_bootstrap(pooled_hits, rrf_hits, n_resamples=10_000, seed=0)
    print(f"paired bootstrap delta (RRF - pooled): {diff_mean:.4f} [{diff_lower:.4f},{diff_upper:.4f}]")
    resolved = not (diff_lower <= 0 <= diff_upper)
    print(f"CI excludes 0 (resolved, not noise-band): {resolved}")

    log_experiment({
        "date": pd.Timestamp.now().isoformat(timespec="seconds"),
        "git_sha": git_sha(),
        "config": f"PAIRED_per-option-RRF-vs-pooled-query_recall@{K_FINAL}_general-corpus",
        "tier": "T1",
        "n": len(t1),
        "map3_mean": round(float(diff_mean), 4),
        "map3_ci_lower": round(float(diff_lower), 4),
        "map3_ci_upper": round(float(diff_upper), 4),
        "random_baseline": "",
        "train_seconds": "",
        "eval_seconds": round(rrf_seconds, 1),
        "hypothesis": (
            "does per-option retrieval (5 queries of prompt+option_i, RRF-fused) beat a single "
            "pooled prompt+all-options query on answer-support recall@5 -- PLAN.md's "
            "'single highest-expected-value trick'"
        ),
        "notes": (
            f"map3_mean/ci hold the PAIRED DELTA in answer-support recall@{K_FINAL} "
            f"(per-option RRF minus pooled), not a MAP@3. Pooled {pooled_hits.mean():.4f} vs "
            f"per-option RRF {rrf_hits.mean():.4f}, same 1500 T1 rows, same general-corpus index "
            f"({len(chunk_texts)} chunks), k_per_query={K_PER_QUERY}, RRF k=60. Retrieval-only "
            f"metric: needs no reader, so unlike every reader-level verdict in "
            f"reports/ablation_table.md it is unaffected by the ln(5) undertrained-reader finding. "
            f"Cost: {rrf_seconds:.1f}s vs {pooled_seconds:.1f}s "
            f"({rrf_seconds / max(pooled_seconds, 1e-9):.1f}x), since it issues 5x the queries."
        ),
    })


if __name__ == "__main__":
    main()
