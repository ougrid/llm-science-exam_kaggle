"""Three-tier eval split construction and text-level leakage checks.

Tiers, per PLAN.md:
    T1 - dev/selection: held-out synthetic rows. Every design decision is made here.
    T2 - gold/confirmation: the 200 official rows. Sacred; never trained on.
    T3 - out-of-distribution: human-written ARC-Challenge + MMLU-STEM.

Leakage control implemented here is text-level only: near-duplicate
question detection via exact character-shingle Jaccard similarity (the
pool sizes here are small enough that an approximate MinHash/LSH index
buys nothing over computing exact Jaccard directly, so that's what this
does — call it what it is).

The article-level contamination audit described in PLAN.md (does a
gold question's retrieved source article also appear in the T1 training
pool?) requires a retriever, which doesn't exist until Day 2/3. It is
NOT implemented here — do not fake it with a stub that always passes.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

OPTION_COLUMNS = ["A", "B", "C", "D", "E"]

# The "STEM" subject grouping used throughout PLAN.md, matching the grouping
# introduced in the original MMLU paper (Hendrycks et al. 2021) and reused by
# lm-evaluation-harness's mmlu_stem task group.
MMLU_STEM_SUBJECTS = [
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "electrical_engineering",
    "elementary_mathematics",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_mathematics",
    "high_school_physics",
    "high_school_statistics",
    "machine_learning",
]


def load_gold(path: str | Path) -> pd.DataFrame:
    """Load the official 200-row train.csv as the T2 gold tier. Never train on this."""
    df = pd.read_csv(path)
    df["tier"] = "T2"
    return df


def load_pool(paths: list[str | Path]) -> pd.DataFrame:
    """Load and concatenate synthetic training pools sharing the prompt/A-E/answer schema."""
    frames = [pd.read_csv(p)[["prompt", *OPTION_COLUMNS, "answer"]] for p in paths]
    return pd.concat(frames, ignore_index=True)


def _char_shingles(text: str, n: int = 5) -> set[str]:
    text = " ".join(str(text).lower().split())
    if len(text) < n:
        return {text}
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def jaccard_similarity(a: str, b: str, n: int = 5) -> float:
    sa, sb = _char_shingles(a, n), _char_shingles(b, n)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def near_duplicate_mask(
    candidates: pd.Series, reference: pd.Series, threshold: float = 0.8, n: int = 5
) -> pd.Series:
    """True for each candidate whose prompt is a near-duplicate of any reference prompt.

    O(len(candidates) * len(reference)) exact Jaccard — fine up to a few
    thousand rows on each side; PLAN.md's tier sizes are well within that.
    """
    ref_shingles = [_char_shingles(t, n) for t in reference]
    flags = []
    for cand in candidates:
        cs = _char_shingles(cand, n)
        is_dup = False
        if cs:
            for rs in ref_shingles:
                if rs and len(cs & rs) / len(cs | rs) >= threshold:
                    is_dup = True
                    break
        flags.append(is_dup)
    return pd.Series(flags, index=candidates.index)


def build_t1(
    pool: pd.DataFrame, gold: pd.DataFrame, n: int = 1500, seed: int = 42
) -> pd.DataFrame:
    """Sample T1 dev rows from a synthetic pool, excluding near-duplicates of gold."""
    dup_of_gold = near_duplicate_mask(pool["prompt"], gold["prompt"])
    clean = pool.loc[~dup_of_gold].drop_duplicates(subset="prompt")
    if len(clean) < n:
        raise ValueError(f"only {len(clean)} clean pool rows available, need {n}")
    t1 = clean.sample(n=n, random_state=seed).reset_index(drop=True)
    t1["tier"] = "T1"
    return t1


def normalize_arc(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize ai2_arc/ARC-Challenge rows to (question, options, answer_letter).

    ARC's answerKey is sometimes a letter and sometimes a digit depending on
    the row; re-derive the correct label by *position* in the choices array
    so it's always a letter, consistent with how the competition itself
    assigns A-E by position rather than by any semantic label.
    """
    records = []
    for _, row in df.iterrows():
        labels = list(row["choices"]["label"])
        texts = list(row["choices"]["text"])
        try:
            correct_idx = labels.index(row["answerKey"])
        except ValueError:
            continue  # malformed row: answerKey not found among its own labels
        records.append(
            {
                "id": row["id"],
                "source": "arc_challenge",
                "subject": None,
                "question": row["question"],
                "options": texts,
                "answer": chr(ord("A") + correct_idx),
            }
        )
    return pd.DataFrame.from_records(records)


def normalize_mmlu(df: pd.DataFrame, subjects: list[str] | None = None) -> pd.DataFrame:
    """Normalize cais/mmlu rows to (question, options, answer_letter), filtered to `subjects`."""
    if subjects is not None:
        df = df[df["subject"].isin(subjects)]
    records = []
    for i, row in df.iterrows():
        records.append(
            {
                "id": f"mmlu_{i}",
                "source": "mmlu",
                "subject": row["subject"],
                "question": row["question"],
                "options": list(row["choices"]),
                "answer": chr(ord("A") + int(row["answer"])),
            }
        )
    return pd.DataFrame.from_records(records)


def build_t3(
    arc_norm: pd.DataFrame, mmlu_norm: pd.DataFrame, n_each: int = 500, seed: int = 42
) -> pd.DataFrame:
    """Sample n_each rows from each normalized OOD source and concatenate.

    MMLU's raw test set contains a small number of exact-duplicate question
    stems (a genuinely repeated item within college_physics, and a generic
    "Which statement is true?" stem that coincidentally matches across two
    different math subjects with different options) — drop duplicates by
    question text, keeping the first occurrence, before sampling.
    """
    arc_clean = arc_norm.drop_duplicates(subset="question")
    mmlu_clean = mmlu_norm.drop_duplicates(subset="question")
    arc_sample = arc_clean.sample(n=min(n_each, len(arc_clean)), random_state=seed)
    mmlu_sample = mmlu_clean.sample(n=min(n_each, len(mmlu_clean)), random_state=seed)
    t3 = pd.concat([arc_sample, mmlu_sample], ignore_index=True)
    t3["tier"] = "T3"
    return t3


def save_t3(t3: pd.DataFrame, path: str | Path) -> None:
    """Save T3 with its variable-length `options` list encoded as JSON text."""
    out = t3.copy()
    out["options"] = out["options"].apply(json.dumps)
    out.to_parquet(path, index=False)


def load_t3(path: str | Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["options"] = df["options"].apply(json.loads)
    return df
