"""Cheap phrase-match rerank: fix BM25's lack of entity/phrase awareness
without a new model.

Motivated directly by hand-inspection (DEVLOG.md): BM25's bag-of-words
scoring ranked a chunk about "Big mama" (a Chinese internet-censorship
term) and an unrelated athlete named "Thornton" above the actual "Big Mama
Thornton" passage, because it scores "Big" + "mama" + "Thornton" as three
independent tokens rather than one entity. This reranks BM25's top-K
candidates by how many multi-word phrases from the query they contain
verbatim, before falling back to the original BM25 score -- a cheap proxy
for entity/phrase awareness that needs no cross-encoder.
"""

from __future__ import annotations

import re

_WORD_RE = re.compile(r"[A-Za-z0-9]+")


def extract_ngrams(text: str, ns: tuple[int, ...] = (2, 3, 4)) -> set[str]:
    """Lowercased word n-grams of the given sizes, space-joined."""
    words = _WORD_RE.findall(text.lower())
    grams: set[str] = set()
    for n in ns:
        for i in range(len(words) - n + 1):
            grams.add(" ".join(words[i : i + n]))
    return grams


def phrase_match_count(query_ngrams: set[str], chunk_text: str) -> int:
    chunk_lower = chunk_text.lower()
    return sum(1 for gram in query_ngrams if gram in chunk_lower)


def rerank_by_phrase_match(
    query: str,
    candidates: list[tuple[int, float]],
    chunk_texts: list[str],
) -> list[tuple[int, float, int]]:
    """Reorder `candidates` (list of (chunk_idx, bm25_score)) by phrase-match
    count first, original BM25 score second. Returns (chunk_idx, bm25_score,
    phrase_match_count) tuples in the new order.
    """
    query_ngrams = extract_ngrams(query)
    scored = [
        (idx, score, phrase_match_count(query_ngrams, chunk_texts[idx])) for idx, score in candidates
    ]
    scored.sort(key=lambda x: (-x[2], -x[1]))
    return scored
