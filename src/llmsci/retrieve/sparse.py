"""BM25 sparse retrieval over the chunk corpus, via bm25s.

Flat (non-hierarchical) index over all chunks. PLAN.md's hierarchical
article->chunk design becomes necessary at the full ~270k-article / ~2M-chunk
scale; the 20k-article pilot slice (168k chunks) is small enough that a flat
BM25 index is fast to build and query directly.
"""

from __future__ import annotations

from pathlib import Path

import bm25s

OPTION_COLUMNS = ["A", "B", "C", "D", "E"]


def build_query(prompt: str, options: list[str]) -> str:
    """PLAN.md's recommended query variant: prompt + all options, space-joined.

    The options act as free query expansion, carrying rare anchor terms the
    prompt alone might not.
    """
    return f"{prompt} " + " ".join(options)


class BM25Index:
    """Wraps bm25s.BM25 with the chunk texts kept alongside for lookup."""

    def __init__(self, texts: list[str]):
        self.texts = texts
        corpus_tokens = bm25s.tokenize(texts, stopwords="en", show_progress=False)
        self.retriever = bm25s.BM25(method="lucene")
        self.retriever.index(corpus_tokens, show_progress=False)

    def search(self, query: str, k: int = 5) -> list[tuple[int, float]]:
        query_tokens = bm25s.tokenize(query, stopwords="en", show_progress=False)
        results, scores = self.retriever.retrieve(query_tokens, k=k, show_progress=False)
        return list(zip(results[0].tolist(), scores[0].tolist()))

    def search_batch(self, queries: list[str], k: int = 5) -> list[list[tuple[int, float]]]:
        query_tokens = bm25s.tokenize(queries, stopwords="en", show_progress=False)
        results, scores = self.retriever.retrieve(query_tokens, k=k, show_progress=False)
        return [list(zip(results[i].tolist(), scores[i].tolist())) for i in range(len(queries))]

    def save(self, path: Path) -> None:
        self.retriever.save(str(path))

    @classmethod
    def load(cls, path: Path, texts: list[str]) -> "BM25Index":
        obj = cls.__new__(cls)
        obj.texts = texts
        obj.retriever = bm25s.BM25.load(str(path))
        return obj
