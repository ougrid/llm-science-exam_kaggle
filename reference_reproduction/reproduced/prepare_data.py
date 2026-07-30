"""Build the train / monitor / T1-eval frames for the cdeotte reproduction.

cdeotte trains on a random 1,024-row sample of `all_12_with_context2.csv` and
validates on the official 200. Two changes, both forced (see ../NOTES.md):

1. LEAKAGE GUARD. Every one of our 1,500 T1 dev prompts is present in
   `all_12_with_context2.csv` -- T1 was built from radek1's pools, which are
   sources 1-6 of cdeotte's 12. A literal `.sample(1024)` would drag ~26 T1
   rows into training and inflate the number this whole track exists to
   report. All T1-matching rows are dropped from the pool before sampling,
   and the script asserts the result.

2. NO GOLD. cdeotte's in-training validation set is the official 200, which
   this project treats as sacred. A held-out slice of the training pool is
   used for the learning curve instead. Because cdeotte sets
   `load_best_model_at_end=False`, nothing is ever selected on it, so this
   changes no result -- it is monitoring only.

T1's context comes from the same file as the training context, i.e. the same
mgoksu retrieval run, which is what makes train/eval retriever consistency
hold by construction.

Writes to reference_reproduction/data/ (gitignored artifacts; nothing outside
reference_reproduction/ is touched).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from common import NUM_TRAIN_SAMPLES, OPTIONS

# The main pipeline's data/ is gitignored, so it exists only in the primary
# working tree. Read-only use.
MAIN_DATA = Path("/home/ougrid/claude-projects_ougridd/llm-science-exam_kaggle/data")
OUT = Path(__file__).resolve().parent.parent / "data"

CDEOTTE_60K = MAIN_DATA / "kaggle_datasets" / "all_12_with_context2.csv"
T1_DEV = MAIN_DATA / "t1_dev.csv"
GOLD = MAIN_DATA / "train.csv"

N_MONITOR = 500
SEED = 42


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-train", type=int, default=NUM_TRAIN_SAMPLES,
                   help="cdeotte's NUM_TRAIN_SAMPLES; his 1024 is explicitly a "
                        "demo subset of the 60k ('TRAIN WITH SUBSET OF 60K')")
    args = p.parse_args()
    n_train = args.n_train
    OUT.mkdir(parents=True, exist_ok=True)

    pool = pd.read_csv(CDEOTTE_60K)
    print(f"cdeotte 60k pool: {pool.shape}")

    # cdeotte's handling of the dataset's missing options: fillna(''), NOT a
    # stringified 'nan'. 22.7% of rows have at least one null option.
    null_rate = pool[OPTIONS].isna().any(axis=1).mean()
    pool[OPTIONS] = pool[OPTIONS].fillna("")
    pool["context"] = pool["context"].fillna("").astype(str)
    print(f"rows with >=1 null option: {null_rate:.1%} (filled with '')")

    t1 = pd.read_csv(T1_DEV)
    gold = pd.read_csv(GOLD)

    t1_prompts = set(t1["prompt"])
    gold_prompts = set(gold["prompt"])

    # --- leakage accounting -------------------------------------------------
    in_t1 = pool["prompt"].isin(t1_prompts)
    in_gold = pool["prompt"].isin(gold_prompts)
    print(f"pool rows matching a T1 dev prompt : {int(in_t1.sum())} "
          f"({pool.loc[in_t1, 'prompt'].nunique()} distinct prompts)")
    print(f"pool rows matching a gold prompt   : {int(in_gold.sum())}")

    # --- T1 eval frame: attach cdeotte's own retrieved context --------------
    ctx = (
        pool.loc[in_t1, ["prompt", "context", "source"]]
        .drop_duplicates(subset="prompt", keep="first")
        .set_index("prompt")
    )
    t1_eval = t1.copy()
    t1_eval["context"] = t1_eval["prompt"].map(ctx["context"])
    t1_eval["ctx_source"] = t1_eval["prompt"].map(ctx["source"])
    missing = int(t1_eval["context"].isna().sum())
    if missing:
        raise RuntimeError(
            f"{missing} T1 rows have no cdeotte context; the join is the whole "
            "basis of this reproduction, refusing to score a partial set"
        )
    print(f"T1 eval frame: {len(t1_eval)} rows, all with cdeotte context")
    print("T1 context source mix:")
    print(t1_eval["ctx_source"].value_counts().sort_index().to_string())

    # --- training pool: drop T1 and gold matches ---------------------------
    train_pool = pool.loc[~in_t1 & ~in_gold].reset_index(drop=True)
    print(f"train pool after dropping T1/gold matches: {len(train_pool)} rows")

    # cdeotte's 60k holds 60,347 rows but only 54,209 distinct prompts -- 6,138
    # duplicate-prompt rows (10.2%). He never trips on this because his
    # validation set is the disjoint official 200; here it would put the same
    # prompt in both train and monitor, so dedupe before splitting. Caught by
    # the train/monitor assertion below, not by inspection.
    before = len(train_pool)
    train_pool = train_pool.drop_duplicates(subset="prompt").reset_index(drop=True)
    print(f"train pool after prompt dedupe: {len(train_pool)} rows "
          f"(-{before - len(train_pool)} duplicate prompts)")

    # Monitor slice is drawn first and held fixed across --n-train settings, so
    # learning curves from different training scales stay comparable.
    sampled = train_pool.sample(n=n_train + N_MONITOR, random_state=SEED)
    monitor_df = sampled.iloc[:N_MONITOR].reset_index(drop=True)
    train_df = sampled.iloc[N_MONITOR:].reset_index(drop=True)

    # --- assertions: the guard has to actually hold ------------------------
    assert not set(train_df["prompt"]) & t1_prompts, "T1 prompt leaked into train"
    assert not set(monitor_df["prompt"]) & t1_prompts, "T1 prompt leaked into monitor"
    assert not set(train_df["prompt"]) & gold_prompts, "gold prompt leaked into train"
    assert not set(train_df["prompt"]) & set(monitor_df["prompt"]), "train/monitor overlap"
    assert train_df["answer"].isin(OPTIONS).all(), "unexpected answer label in train"
    assert t1_eval["answer"].isin(OPTIONS).all(), "unexpected answer label in T1"

    print(f"train: {len(train_df)} rows  (cdeotte's NUM_TRAIN_SAMPLES default = {NUM_TRAIN_SAMPLES})")
    print("train context source mix:")
    print(train_df["source"].value_counts().sort_index().to_string())
    print(f"monitor: {len(monitor_df)} rows (replaces cdeotte's gold-200 validation set)")

    train_df.to_parquet(OUT / f"train_{n_train}.parquet", index=False)
    monitor_df.to_parquet(OUT / "monitor_500.parquet", index=False)
    t1_eval.to_parquet(OUT / "t1_eval_cdeotte_ctx.parquet", index=False)
    print(f"wrote 3 frames to {OUT}")


if __name__ == "__main__":
    main()
