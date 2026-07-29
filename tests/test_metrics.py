import numpy as np
import pytest

from llmsci.metrics import (
    average_precision_at_k,
    average_precision_scores,
    bootstrap_ci,
    map_at_k,
    minimum_detectable_effect,
    paired_bootstrap,
    random_baseline_map_at_k,
)


def test_ap_rank_1():
    assert average_precision_at_k("A", ["A", "B", "C"]) == 1.0


def test_ap_rank_2():
    assert average_precision_at_k("B", ["A", "B", "C"]) == pytest.approx(0.5)


def test_ap_rank_3():
    assert average_precision_at_k("C", ["A", "B", "C"]) == pytest.approx(1 / 3)


def test_ap_absent():
    assert average_precision_at_k("D", ["A", "B", "C"]) == 0.0


def test_ap_beyond_k_counts_as_absent():
    assert average_precision_at_k("D", ["A", "B", "C", "D"]) == 0.0


def test_ap_duplicate_prediction_gets_no_second_chance():
    assert average_precision_at_k("A", ["A", "A", "B"]) == 1.0
    assert average_precision_at_k("C", ["A", "A", "B"]) == 0.0


def test_map_at_k_averages_rows():
    y_true = ["A", "B", "C", "D"]
    y_pred = [["A", "B", "C"]] * 4
    expected = (1.0 + 0.5 + 1 / 3 + 0.0) / 4
    assert map_at_k(y_true, y_pred) == pytest.approx(expected)


def test_map_at_k_mismatched_lengths_raises():
    with pytest.raises(ValueError):
        map_at_k(["A", "B"], [["A", "B", "C"]])


def test_random_baseline_is_0_3667_for_5_options_k3():
    assert random_baseline_map_at_k(num_options=5, k=3) == pytest.approx(11 / 30)
    assert random_baseline_map_at_k(num_options=5, k=3) == pytest.approx(0.3667, abs=1e-4)


def test_random_submission_lands_within_its_own_bootstrap_ci():
    # A genuinely random 3-of-5 ranking against a uniform true label should
    # be consistent with the analytic 0.3667 -- the sanity check PLAN.md
    # calls for before trusting any real submission's score.
    rng = np.random.default_rng(42)
    n = 4000
    options = list("ABCDE")
    y_true = rng.choice(options, size=n).tolist()
    y_pred = [rng.permutation(options)[:3].tolist() for _ in range(n)]
    scores = average_precision_scores(y_true, y_pred)
    _, lower, upper = bootstrap_ci(scores, n_resamples=2000, seed=1)
    assert lower <= random_baseline_map_at_k() <= upper


def test_bootstrap_ci_bounds_contain_mean():
    scores = np.array([1.0, 0.5, 0.0, 1 / 3, 0.0, 1.0, 0.5])
    mean, lower, upper = bootstrap_ci(scores, n_resamples=2000, seed=0)
    assert lower <= mean <= upper


def test_bootstrap_ci_empty_raises():
    with pytest.raises(ValueError):
        bootstrap_ci(np.array([]))


def test_paired_bootstrap_identical_scores_ci_includes_zero():
    scores = np.array([1.0, 0.5, 0.0, 1 / 3, 1.0])
    mean, lower, upper = paired_bootstrap(scores, scores.copy(), n_resamples=2000, seed=0)
    assert mean == 0.0
    assert lower <= 0.0 <= upper


def test_paired_bootstrap_clear_improvement_excludes_zero():
    rng = np.random.default_rng(0)
    n = 2000
    scores_a = rng.choice([0.0, 1 / 3, 0.5, 1.0], size=n, p=[0.4, 0.2, 0.2, 0.2])
    scores_b = rng.choice([0.0, 1 / 3, 0.5, 1.0], size=n, p=[0.1, 0.1, 0.1, 0.7])
    mean, lower, upper = paired_bootstrap(scores_a, scores_b, n_resamples=2000, seed=1)
    assert mean > 0
    assert lower > 0  # CI excludes 0: a real, resolvable improvement


def test_paired_bootstrap_mismatched_lengths_raises():
    with pytest.raises(ValueError):
        paired_bootstrap(np.array([1.0, 0.0]), np.array([1.0]))


def test_minimum_detectable_effect_positive_and_shrinks_with_n():
    rng = np.random.default_rng(0)
    small_a = rng.choice([0.0, 1.0], size=50).astype(float)
    small_b = rng.choice([0.0, 1.0], size=50).astype(float)
    large_a = rng.choice([0.0, 1.0], size=5000).astype(float)
    large_b = rng.choice([0.0, 1.0], size=5000).astype(float)
    mde_small = minimum_detectable_effect(small_a, small_b)
    mde_large = minimum_detectable_effect(large_a, large_b)
    assert mde_small > 0
    assert mde_large > 0
    assert mde_large < mde_small  # more rows -> tighter resolution


def test_minimum_detectable_effect_unknown_ci_raises():
    a = np.array([0.0, 1.0, 0.5])
    b = np.array([1.0, 0.0, 0.5])
    with pytest.raises(ValueError):
        minimum_detectable_effect(a, b, ci=0.42)
