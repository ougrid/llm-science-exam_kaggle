"""Reciprocal Rank Fusion, and per-option retrieval built on top of it.

PLAN.md calls per-option retrieval "the single highest-expected-value trick on
the list", for a mechanistic reason: a single pooled query
(`prompt + all five options`) mixes the correct answer's rare anchor terms
with four distractors' terms, and distractors frequently live in *different*
Wikipedia articles than the answer. A pooled query therefore systematically
under-retrieves the evidence needed to *reject* options. Running one query per
option and fusing the five ranked lists retrieves evidence for each option on
its own terms -- it is also the retrieval-side mirror of 1st place's
per-option contextualization.

RRF is used rather than score averaging because BM25 scores are not
comparable across queries of different lengths: each per-option query has a
different term composition, so its score scale differs. Rank is invariant to
that. k=60 is the standard constant from Cormack et al.'s original RRF paper
and the value PLAN.md specifies.
"""

from __future__ import annotations

from llmsci.retrieve.sparse import BM25Index

RRF_K = 60


def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[int, float]]], k: int = RRF_K
) -> list[tuple[int, float]]:
    """Fuse ranked lists of (doc_id, score) into one, by RRF.

    Each list contributes `1 / (k + rank)` per document, rank being 1-indexed
    within that list. The original scores are deliberately ignored -- only
    rank order carries across lists (see module docstring).
    """
    fused: dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, (doc_id, _score) in enumerate(ranked, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda kv: -kv[1])


def per_option_queries(prompt: str, options: list[str]) -> list[str]:
    """One query per option: `prompt + " " + option_i`.

    Contrast with `sparse.build_query`, which concatenates the prompt and all
    five options into a single query.
    """
    return [f"{prompt} {opt}" for opt in options]


def search_per_option_rrf(
    index: BM25Index,
    prompts: list[str],
    options_per_row: list[list[str]],
    k_per_query: int = 20,
    k_final: int = 5,
    rrf_k: int = RRF_K,
) -> list[list[tuple[int, float]]]:
    """Per-option retrieval + RRF fusion, batched over rows.

    Issues `5 * len(prompts)` queries in one `search_batch` call rather than
    five separate calls, so the underlying bm25s scoring stays vectorized.
    Returns the fused top-`k_final` per row; the returned float is the RRF
    score, not a BM25 score, and is only meaningful for ranking.
    """
    flat_queries: list[str] = []
    for prompt, options in zip(prompts, options_per_row):
        flat_queries.extend(per_option_queries(prompt, options))

    flat_results = index.search_batch(flat_queries, k=k_per_query)

    fused_per_row: list[list[tuple[int, float]]] = []
    cursor = 0
    for options in options_per_row:
        n = len(options)
        fused = reciprocal_rank_fusion(flat_results[cursor : cursor + n], k=rrf_k)
        fused_per_row.append(fused[:k_final])
        cursor += n
    return fused_per_row
