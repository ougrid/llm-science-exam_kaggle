from pathlib import Path

import pandas as pd
import pytest

from llmsci.data import build_t1, jaccard_similarity, near_duplicate_mask

DATA = Path(__file__).resolve().parents[1] / "data"


def test_jaccard_similarity_identical_text_is_1():
    assert jaccard_similarity("What is the capital of France?", "What is the capital of France?") == 1.0


def test_jaccard_similarity_unrelated_text_is_low():
    a = "What is the capital of France?"
    b = "Explain the mechanism of nuclear fission in heavy elements."
    assert jaccard_similarity(a, b) < 0.2


def test_jaccard_similarity_near_duplicate_with_minor_edit_is_high():
    a = "Which of the following best describes Modified Newtonian Dynamics?"
    b = "Which of the following best describes modified newtonian dynamics!"
    assert jaccard_similarity(a, b) > 0.8


def test_near_duplicate_mask_flags_only_the_duplicate():
    candidates = pd.Series(
        [
            "Which of the following best describes Modified Newtonian Dynamics?",
            "What is the boiling point of water at sea level?",
        ]
    )
    reference = pd.Series(["Which of the following best describes modified newtonian dynamics!"])
    mask = near_duplicate_mask(candidates, reference, threshold=0.8)
    assert mask.tolist() == [True, False]


def test_build_t1_excludes_near_duplicates_of_gold():
    gold = pd.DataFrame({"prompt": ["Unique gold question about thermodynamics."]})
    pool = pd.DataFrame(
        {
            "prompt": [
                "Unique gold question about thermodynamics!",  # near-dup of gold, must be excluded
                "A completely different question about cell biology.",
                "Another distinct question about orbital mechanics.",
            ],
            "A": ["a", "a", "a"],
            "B": ["b", "b", "b"],
            "C": ["c", "c", "c"],
            "D": ["d", "d", "d"],
            "E": ["e", "e", "e"],
            "answer": ["A", "A", "A"],
        }
    )
    t1 = build_t1(pool, gold, n=2, seed=0)
    assert len(t1) == 2
    assert "thermodynamics!" not in t1["prompt"].tolist()[0]
    assert not any(jaccard_similarity(p, gold["prompt"].iloc[0]) >= 0.8 for p in t1["prompt"])


def test_build_t1_raises_if_pool_too_small_after_dedup():
    gold = pd.DataFrame({"prompt": ["x"]})
    pool = pd.DataFrame(
        {
            "prompt": ["only one clean row"],
            "A": ["a"], "B": ["b"], "C": ["c"], "D": ["d"], "E": ["e"], "answer": ["A"],
        }
    )
    with pytest.raises(ValueError):
        build_t1(pool, gold, n=5)


# --- Integration checks against the materialized tiers on this machine ---
# Skip cleanly if scripts/build_eval_tiers.py hasn't been run yet (e.g. fresh
# clone, or CI without the Kaggle-downloaded pools) rather than erroring.

_gold_path = DATA / "holdout_gold.csv"
_t1_path = DATA / "t1_dev.csv"
_t3_path = DATA / "t3_ood.parquet"
_tiers_built = _gold_path.exists() and _t1_path.exists() and _t3_path.exists()

pytestmark_tiers = pytest.mark.skipif(
    not _tiers_built, reason="eval tiers not built yet — run scripts/build_eval_tiers.py"
)


@pytestmark_tiers
def test_t1_has_no_near_duplicates_of_gold():
    gold = pd.read_csv(_gold_path)
    t1 = pd.read_csv(_t1_path)
    dup_mask = near_duplicate_mask(t1["prompt"], gold["prompt"])
    assert not dup_mask.any(), f"{dup_mask.sum()} T1 rows are near-duplicates of a gold row"


@pytestmark_tiers
def test_t1_has_no_internal_exact_duplicates():
    t1 = pd.read_csv(_t1_path)
    assert t1["prompt"].duplicated().sum() == 0


@pytestmark_tiers
def test_gold_has_no_internal_exact_duplicates():
    gold = pd.read_csv(_gold_path)
    assert gold["prompt"].duplicated().sum() == 0


@pytestmark_tiers
def test_t3_has_no_internal_exact_duplicate_questions():
    t3 = pd.read_parquet(_t3_path)
    assert t3["question"].duplicated().sum() == 0
