import pandas as pd

from llmsci.retrieve.eval import (
    distinctive_keywords,
    is_answer_support_hit,
    mrr,
    ndcg_at_k,
    recall_at_k,
)


def test_distinctive_keywords_excludes_shared_words():
    row = pd.Series(
        {
            "A": "photosynthesis converts sunlight into chemical energy",
            "B": "photosynthesis converts sunlight into thermal energy",
            "C": "unrelated distractor about mitochondria",
            "D": "another unrelated distractor",
            "E": "yet another distractor option",
            "answer": "A",
        }
    )
    kw = distinctive_keywords(row)
    assert "chemical" in kw
    assert "photosynthesis" not in kw  # shared with B
    assert "sunlight" not in kw  # shared with B


def test_is_answer_support_hit_true_and_false():
    kw = {"chemical", "photosynthesis"}
    assert is_answer_support_hit("This passage discusses chemical bonds in plants.", kw)
    assert not is_answer_support_hit("This passage is about something else entirely.", kw)


def test_is_answer_support_hit_empty_keywords_is_never_a_hit():
    assert not is_answer_support_hit("chemical energy is discussed here", set())


def test_recall_at_k_thresholds_correctly():
    ranks = [1, 3, 0, 10, 50]
    assert recall_at_k(ranks, 1).tolist() == [1.0, 0.0, 0.0, 0.0, 0.0]
    assert recall_at_k(ranks, 5).tolist() == [1.0, 1.0, 0.0, 0.0, 0.0]
    assert recall_at_k(ranks, 50).tolist() == [1.0, 1.0, 0.0, 1.0, 1.0]


def test_mrr_reciprocal_and_zero_for_miss():
    ranks = [1, 2, 0, 4]
    result = mrr(ranks)
    assert result.tolist() == [1.0, 0.5, 0.0, 0.25]


def test_ndcg_at_k_matches_reciprocal_log_and_respects_k():
    ranks = [1, 3, 0]
    result = ndcg_at_k(ranks, k=2)
    assert result[0] == 1.0  # log2(2) = 1
    assert result[1] == 0.0  # rank 3 > k=2
    assert result[2] == 0.0  # miss
