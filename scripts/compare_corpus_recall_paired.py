"""Paired comparison of retrieval recall: mbanaei (STEM-only) vs. the general
Wikipedia corpus, on the SAME T1 rows -- the proper version of the
run_retrieval_eval.py / run_retrieval_eval_general.py comparison, which
reports two separate CIs (exactly what CLAUDE.md's non-negotiable rules
warn against comparing directly).

Run once from the repo root, after both scripts/build_bm25_index_full.py and
scripts/build_bm25_index_general.py: `python scripts/compare_corpus_recall_paired.py`
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from llmsci.experiment import git_sha, log_experiment
from llmsci.metrics import paired_bootstrap
from llmsci.retrieve.eval import recall_at_k, retrieve_ranks
from llmsci.retrieve.sparse import BM25Index

DATA = Path("data")
STEM_INDEX_DIR = DATA / "bm25_index_full"
GENERAL_INDEX_DIR = DATA / "bm25_index_general"
K = 5


def load_index(index_dir: Path) -> tuple[BM25Index, list[str]]:
    chunks = pd.read_parquet(index_dir / "chunk_texts.parquet")
    chunk_texts = chunks["text"].tolist()
    return BM25Index.load(index_dir, chunk_texts), chunk_texts


def main() -> None:
    t1 = pd.read_csv(DATA / "t1_dev.csv")

    stem_index, stem_texts = load_index(STEM_INDEX_DIR)
    stem_ranks = retrieve_ranks(t1, stem_index, stem_texts, max_k=K)
    stem_scores = recall_at_k(stem_ranks, K)

    general_index, general_texts = load_index(GENERAL_INDEX_DIR)
    general_ranks = retrieve_ranks(t1, general_index, general_texts, max_k=K)
    general_scores = recall_at_k(general_ranks, K)

    print(f"mbanaei (STEM-only, full corpus) recall@{K} mean: {stem_scores.mean():.4f}")
    print(f"general Wikipedia (1.6M-chunk sample) recall@{K} mean: {general_scores.mean():.4f}")

    diff_mean, diff_lower, diff_upper = paired_bootstrap(stem_scores, general_scores, n_resamples=10_000, seed=0)
    print(f"paired bootstrap delta (general - mbanaei): {diff_mean:.4f} [{diff_lower:.4f},{diff_upper:.4f}]")
    resolved = not (diff_lower <= 0 <= diff_upper)
    print(f"CI excludes 0 (resolved, not noise-band): {resolved}")

    log_experiment(
        {
            "date": pd.Timestamp.now().isoformat(timespec="seconds"),
            "git_sha": git_sha(),
            "config": f"PAIRED_corpus-swap_recall@{K}_general-vs-mbanaei-STEM",
            "tier": "T1",
            "n": len(t1),
            "map3_mean": round(float(diff_mean), 4),
            "map3_ci_lower": round(float(diff_lower), 4),
            "map3_ci_upper": round(float(diff_upper), 4),
            "random_baseline": "",
            "train_seconds": "",
            "eval_seconds": "",
            "hypothesis": (
                "does swapping mbanaei's STEM-only corpus for a general (non-STEM-filtered) Wikipedia "
                "corpus raise answer-support recall@5 on the SAME T1 rows, measured as a paired delta "
                "rather than two separate CIs (which run_retrieval_eval.py vs "
                "run_retrieval_eval_general.py would otherwise be comparing)"
            ),
            "notes": (
                f"map3_mean/ci columns hold the PAIRED DELTA (general minus mbanaei) in answer-support "
                f"recall@{K}, not a MAP@3 score. mbanaei full corpus (2,101,279 chunks) mean "
                f"{stem_scores.mean():.4f}; general Wikipedia corpus (1,600,063-chunk sample, "
                f"see scripts/build_corpus_general.py) mean {general_scores.mean():.4f}. Both indexes "
                f"evaluated on the identical 1500 T1 rows with the identical query/proxy logic -- only "
                f"the corpus differs. Notably the general corpus is SMALLER (1.6M vs 2.1M chunks) yet "
                f"still {'wins' if diff_mean > 0 else 'loses'}, consistent with the hand-inspection "
                f"finding that the STEM-only corpus structurally cannot contain some entities T1 "
                f"questions reference (e.g. Big Mama Thornton, Didymogenes) regardless of ranking "
                f"quality."
            ),
        }
    )


if __name__ == "__main__":
    main()
