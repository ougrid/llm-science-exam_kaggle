"""Tests for RRF and per-option query construction."""

from llmsci.retrieve.fuse import RRF_K as RRF_K_DEFAULT
from llmsci.retrieve.fuse import per_option_queries, reciprocal_rank_fusion


def test_rrf_single_list_preserves_order():
    fused = reciprocal_rank_fusion([[(7, 9.0), (3, 5.0), (1, 1.0)]])
    assert [doc for doc, _ in fused] == [7, 3, 1]


def test_rrf_rewards_documents_appearing_in_multiple_lists():
    # doc 2 is rank-2 in both lists; docs 1 and 3 are rank-1 in one list each.
    # 2/(60+2) = 0.03226 beats 1/(60+1) = 0.01639, so doc 2 must win.
    fused = reciprocal_rank_fusion([[(1, 9.0), (2, 8.0)], [(3, 9.0), (2, 8.0)]])
    assert fused[0][0] == 2


def test_rrf_ignores_raw_score_magnitude():
    # Identical rank structure, wildly different scales -> identical fusion.
    a = reciprocal_rank_fusion([[(1, 1000.0), (2, 999.0)]])
    b = reciprocal_rank_fusion([[(1, 0.002), (2, 0.001)]])
    assert [d for d, _ in a] == [d for d, _ in b]
    assert [round(s, 12) for _, s in a] == [round(s, 12) for _, s in b]


def test_rrf_k_trades_off_multi_list_agreement_against_top_rank():
    """`k` sets how much "appears in several lists" outweighs "ranked first once".

    doc 1 is rank-1 in one list; doc 2 is rank-2 in both. At k=0 those are
    exactly equal (1/1 == 1/2 + 1/2), so doc 2 does *not* win -- the tie falls
    to insertion order. Raising k discounts the rank-1 advantage faster than
    the agreement bonus, so doc 2 wins from k>=1 onward. Verified by hand
    rather than assumed: an earlier version of this test asserted doc 2 wins
    at k=0 and was simply wrong about the arithmetic.
    """
    lists = [[(1, 9.0), (2, 8.0)], [(3, 9.0), (2, 8.0)]]

    at_zero = dict(reciprocal_rank_fusion(lists, k=0))
    assert at_zero[1] == at_zero[2] == 1.0

    for k in (1, RRF_K_DEFAULT, 10_000):
        assert reciprocal_rank_fusion(lists, k=k)[0][0] == 2


def test_per_option_queries_shape_and_content():
    qs = per_option_queries("what is X?", ["a", "b", "c", "d", "e"])
    assert len(qs) == 5
    assert qs[0] == "what is X? a"
    assert all(q.startswith("what is X?") for q in qs)
