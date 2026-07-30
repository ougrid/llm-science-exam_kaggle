"""Retrieval eval harness: measure the retriever independently of the reader.

Neither T1 nor T3 nor the gold 200 carries a "generating article" label (the
public synthetic pools only ship prompt/options/answer -- checked directly
against radek1's pool, `extra_train_set.csv`, and `t3_ood.parquet` before
writing this), so PLAN.md's "Proxy A: source-article recall" isn't
computable as originally specified. This implements **Proxy B: answer-support
recall** instead (the DPR paper's answer-string recall protocol, adapted for
multiple choice) -- it needs no source-article ground truth and works on
every tier including the gold 200, which is exactly why PLAN.md calls it
out as usable "even on the gold 200, which have no gold passage."

A chunk counts as a hit for a question if it contains at least one content
word that is "rare" relative to the choice set: present in the correct
option's text but in none of the four distractors. This is a proxy, not
ground truth -- it will over-count (a chunk can contain the right words
without supporting the actual inference) and under-count (correct chunks
that paraphrase rather than restate). Both directions are named explicitly
in the numbers this module reports, not smoothed over.
"""

from __future__ import annotations

import re
from typing import Callable

import numpy as np
import pandas as pd

from llmsci.metrics import bootstrap_ci
from llmsci.retrieve.sparse import BM25Index, build_query

OPTION_COLUMNS = ["A", "B", "C", "D", "E"]
_WORD_RE = re.compile(r"[A-Za-z]{5,}")


def distinctive_keywords(row: pd.Series, option_columns: list[str] = OPTION_COLUMNS) -> set[str]:
    """Content words (len>=5) in the correct option not shared by any distractor."""
    answer_col = row["answer"]
    correct_words = set(_WORD_RE.findall(str(row[answer_col]).lower()))
    distractor_words: set[str] = set()
    for c in option_columns:
        if c != answer_col:
            distractor_words |= set(_WORD_RE.findall(str(row[c]).lower()))
    return correct_words - distractor_words


def is_answer_support_hit(chunk_text: str, keywords: set[str]) -> bool:
    if not keywords:
        return False
    chunk_lower = chunk_text.lower()
    return any(kw in chunk_lower for kw in keywords)


def retrieve_ranks(
    df: pd.DataFrame,
    index: BM25Index,
    chunk_texts: list[str],
    max_k: int,
    query_fn: Callable[[str, list[str]], str] = build_query,
    option_columns: list[str] = OPTION_COLUMNS,
) -> list[int]:
    """For each row, the 1-indexed rank of the first answer-support hit among
    the top `max_k` retrieved chunks, or 0 if none of the top `max_k` hit.
    """
    queries = [query_fn(row["prompt"] if "prompt" in row else row["question"], [row[c] for c in option_columns]) for _, row in df.iterrows()]
    results = index.search_batch(queries, k=max_k)
    ranks = []
    for (_, row), row_results in zip(df.iterrows(), results):
        keywords = distinctive_keywords(row, option_columns)
        rank = 0
        for i, (chunk_idx, _score) in enumerate(row_results):
            if is_answer_support_hit(chunk_texts[chunk_idx], keywords):
                rank = i + 1
                break
        ranks.append(rank)
    return ranks


def recall_at_k(ranks: list[int], k: int) -> np.ndarray:
    """Per-row binary hit indicator: was there a hit within the top k."""
    return np.array([1.0 if 0 < r <= k else 0.0 for r in ranks])


def mrr(ranks: list[int]) -> np.ndarray:
    """Per-row reciprocal rank (0 if no hit)."""
    return np.array([1.0 / r if r > 0 else 0.0 for r in ranks])


def ndcg_at_k(ranks: list[int], k: int) -> np.ndarray:
    """Per-row nDCG@k for a single relevant item: DCG = 1/log2(rank+1), IDCG = 1."""
    return np.array([1.0 / np.log2(r + 1) if 0 < r <= k else 0.0 for r in ranks])


def evaluate_retrieval(
    df: pd.DataFrame,
    index: BM25Index,
    chunk_texts: list[str],
    ks: tuple[int, ...] = (1, 5, 10, 20, 50, 100),
    n_resamples: int = 10_000,
    seed: int = 0,
) -> pd.DataFrame:
    """Recall@k (each with a 95% bootstrap CI), MRR, and nDCG@10 for `df` against `index`.

    Returns a small DataFrame, one row per metric, ready to print or save.
    """
    max_k = max(ks)
    ranks = retrieve_ranks(df, index, chunk_texts, max_k)

    rows = []
    for k in ks:
        scores = recall_at_k(ranks, k)
        mean, lower, upper = bootstrap_ci(scores, n_resamples=n_resamples, seed=seed)
        rows.append({"metric": f"recall@{k}", "mean": mean, "ci_lower": lower, "ci_upper": upper})

    mrr_scores = mrr(ranks)
    mean, lower, upper = bootstrap_ci(mrr_scores, n_resamples=n_resamples, seed=seed)
    rows.append({"metric": "MRR", "mean": mean, "ci_lower": lower, "ci_upper": upper})

    ndcg_scores = ndcg_at_k(ranks, 10)
    mean, lower, upper = bootstrap_ci(ndcg_scores, n_resamples=n_resamples, seed=seed)
    rows.append({"metric": "nDCG@10", "mean": mean, "ci_lower": lower, "ci_upper": upper})

    return pd.DataFrame(rows)
