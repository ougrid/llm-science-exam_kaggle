"""MAP@3 and bootstrap comparison utilities.

Single source of truth for scoring predictions in this project. No other
file should recompute MAP@3 or a confidence interval inline — see
CLAUDE.md's "No naked MAP@3 numbers" and "Comparisons use paired
bootstrap" rules.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

_Z_BY_CI = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}


def average_precision_at_k(actual: str, predicted: Sequence[str], k: int = 3) -> float:
    """AP@k for one row with exactly one correct label.

    A correct label at 1-indexed rank r scores 1/r; beyond position k, or
    absent, scores 0. A repeated label in `predicted` cannot score twice.
    """
    seen: set[str] = set()
    for i, p in enumerate(predicted[:k]):
        if p == actual and p not in seen:
            return 1.0 / (i + 1)
        seen.add(p)
    return 0.0


def average_precision_scores(
    y_true: Sequence[str], y_pred: Sequence[Sequence[str]], k: int = 3
) -> np.ndarray:
    """Per-row AP@k array — the input every other function here consumes."""
    if len(y_true) != len(y_pred):
        raise ValueError(f"y_true has {len(y_true)} rows but y_pred has {len(y_pred)}")
    return np.array([average_precision_at_k(t, p, k=k) for t, p in zip(y_true, y_pred)])


def map_at_k(y_true: Sequence[str], y_pred: Sequence[Sequence[str]], k: int = 3) -> float:
    """Mean average precision at k — the competition's leaderboard metric."""
    return float(average_precision_scores(y_true, y_pred, k=k).mean())


def bootstrap_ci(
    scores: np.ndarray,
    n_resamples: int = 10_000,
    ci: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI on the mean of `scores`. Returns (mean, lower, upper)."""
    if len(scores) == 0:
        raise ValueError("cannot bootstrap an empty score array")
    rng = np.random.default_rng(seed)
    n = len(scores)
    idx = rng.integers(0, n, size=(n_resamples, n))
    resample_means = scores[idx].mean(axis=1)
    alpha = (1 - ci) / 2
    lower, upper = np.quantile(resample_means, [alpha, 1 - alpha])
    return float(scores.mean()), float(lower), float(upper)


def paired_bootstrap(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    n_resamples: int = 10_000,
    ci: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Paired bootstrap CI on mean(scores_b - scores_a) for the same rows.

    Accept a change only if this CI excludes 0.
    """
    if len(scores_a) != len(scores_b):
        raise ValueError("paired_bootstrap requires equal-length, row-aligned arrays")
    return bootstrap_ci(scores_b - scores_a, n_resamples=n_resamples, ci=ci, seed=seed)


def minimum_detectable_effect(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    ci: float = 0.95,
) -> float:
    """Smallest mean difference this eval set could plausibly resolve.

    Analytic standard error of the paired per-row difference, so a null
    result can be reported as "not resolved by my eval" rather than
    silently treated as "no effect."
    """
    if len(scores_a) != len(scores_b):
        raise ValueError("minimum_detectable_effect requires equal-length, row-aligned arrays")
    z = _Z_BY_CI.get(ci)
    if z is None:
        raise ValueError(f"no z-value tabulated for ci={ci}; add one to _Z_BY_CI")
    diff = scores_b - scores_a
    se = diff.std(ddof=1) / np.sqrt(len(diff))
    return float(z * se)


def random_baseline_map_at_k(num_options: int = 5, k: int = 3) -> float:
    """Analytic expected MAP@k for a uniformly random ranking (0.3667 for 5 options, k=3)."""
    return sum(1.0 / r for r in range(1, k + 1)) / num_options
