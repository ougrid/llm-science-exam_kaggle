from llmsci.retrieve.rerank import extract_ngrams, phrase_match_count, rerank_by_phrase_match


def test_extract_ngrams_captures_multiword_phrases():
    grams = extract_ngrams("Big Mama Thornton sang the blues")
    assert "big mama" in grams
    assert "big mama thornton" in grams
    assert "mama thornton sang" in grams


def test_phrase_match_count_scores_exact_entity_higher_than_partial():
    query_grams = extract_ngrams("What is Big Mama Thornton's claim to fame?")
    exact = "Blues singer Big Mama Thornton had a hit with Hound Dog."
    unrelated = "Big cats and mama bears are unrelated to Thornton the athlete."
    assert phrase_match_count(query_grams, exact) > phrase_match_count(query_grams, unrelated)


def test_rerank_by_phrase_match_promotes_exact_entity_match():
    query = "What is Big Mama Thornton's claim to fame in music?"
    chunk_texts = [
        "Big cats and mama bears are unrelated to Thornton the athlete.",  # idx 0: partial-token match only
        "Blues singer Big Mama Thornton had a hit with Hound Dog.",  # idx 1: exact entity match
        "Completely unrelated text about geology.",  # idx 2
    ]
    # BM25 (simulated) originally ranked idx 0 above idx 1 despite idx 1 being the real hit
    candidates = [(0, 5.0), (1, 3.0), (2, 1.0)]
    reranked = rerank_by_phrase_match(query, candidates, chunk_texts)
    assert reranked[0][0] == 1  # the exact-entity chunk should now be first


def test_rerank_by_phrase_match_falls_back_to_bm25_score_on_ties():
    query = "some query with no matching phrases at all"
    chunk_texts = ["unrelated a", "unrelated b"]
    candidates = [(0, 1.0), (1, 2.0)]
    reranked = rerank_by_phrase_match(query, candidates, chunk_texts)
    assert reranked[0][0] == 1  # higher original BM25 score wins when phrase-match count ties at 0
