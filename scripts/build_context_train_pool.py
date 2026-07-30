"""Build the open-book training pool from cdeotte's pre-retrieved context dataset.

`data/kaggle_datasets/all_12_with_context2.csv` already has a retrieved
context column per row, letting us skip building our own
retrieval-over-training-data pipeline for the first open-book training run
(PLAN.md Day 2 item 4). Confirmed by inspection before this script was
written:
- 0 of its 60,347 rows overlap `data/train.csv` (the official gold 200) --
  safe to use for training.
- 1,500 rows exact-match (prompt + all five options + answer) a T1 dev row
  -- these must be excluded from training, or T1 leaks into open-book
  training exactly like it would have for the closed-book baseline. Joining
  on `prompt` alone is NOT safe here: 9 prompts in this file are shared by
  multiple rows with *different* options/answers (duplicate question text
  from different source generations), so a prompt-only join produces
  ambiguous many-to-many matches. The full-row join is exact and produces
  a clean 1:1,500 match with zero ambiguity.

This also writes `data/t1_dev_context.parquet`: the same 1,500 T1 rows,
with cdeotte's retrieved context attached, for evaluating the open-book
reader with context from the *same retriever* it was trained on (the
non-negotiable rule in CLAUDE.md) before our own retrieval pipeline exists.

Also drops rows with a NULL option (A-E). Found the hard way: an open-book
training run on the unfiltered pool showed zero learning signal across a
full epoch (loss flat at ln(5), MAP@3 never left the noise band around
baseline) -- 25.8% of `all_12_with_context2.csv`'s rows (13,674/52,923) have
at least one null option, which pandas/f-string stringifies to the literal
text "nan" as a fake answer choice. T1 itself was unaffected (T1's own
options are always non-null, so the exact-row join with `df` can only ever
match `df` rows that also have non-null options there), which is why this
wasn't caught until inspecting the *training* data directly. See
DEVLOG.md's "the vanishing option" entry.

Run once from the repo root: `python scripts/build_context_train_pool.py`
"""

from pathlib import Path

import pandas as pd

DATA = Path("data")
JOIN_COLS = ["prompt", "A", "B", "C", "D", "E", "answer"]


def main() -> None:
    df = pd.read_csv(DATA / "kaggle_datasets/all_12_with_context2.csv")
    gold = pd.read_csv(DATA / "train.csv")
    t1 = pd.read_csv(DATA / "t1_dev.csv")

    gold_overlap = int(df["prompt"].isin(gold["prompt"]).sum())
    assert gold_overlap == 0, f"{gold_overlap} rows overlap the gold 200 -- must not train on these"

    has_null_option = df[["A", "B", "C", "D", "E"]].isna().any(axis=1)
    print(f"dropping {int(has_null_option.sum())}/{len(df)} rows with a null option (stringifies to 'nan')")
    df = df.loc[~has_null_option].reset_index(drop=True)

    t1_context = t1.merge(df, on=JOIN_COLS, how="left")
    missing = int(t1_context["context"].isna().sum())
    assert missing == 0, f"{missing} T1 rows had no exact-match context row"
    assert t1_context["prompt"].duplicated().sum() == 0
    t1_context.to_parquet(DATA / "t1_dev_context.parquet", index=False)
    print(f"t1_dev_context: {len(t1_context)} rows -> data/t1_dev_context.parquet")

    before = len(df)
    is_t1_row = df.set_index(JOIN_COLS).index.isin(t1.set_index(JOIN_COLS).index)
    train_pool_context = df.loc[~is_t1_row].drop_duplicates(subset=JOIN_COLS).reset_index(drop=True)
    print(f"all_12_with_context2: {before} rows")
    print(f"removed {before - len(train_pool_context)} rows (T1 exact-match + full-row exact duplicates)")
    print(f"train_pool_context: {len(train_pool_context)} rows -> data/train_pool_context.parquet")

    train_pool_context.to_parquet(DATA / "train_pool_context.parquet", index=False)


if __name__ == "__main__":
    main()
