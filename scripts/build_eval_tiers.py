"""Build and save the three eval tiers (T1 dev, T2 gold, T3 OOD) into data/.

Run once from the repo root: `python scripts/build_eval_tiers.py`
"""

from pathlib import Path

import pandas as pd

from llmsci.data import (
    MMLU_STEM_SUBJECTS,
    build_t1,
    build_t3,
    load_gold,
    load_pool,
    near_duplicate_mask,
    normalize_arc,
    normalize_mmlu,
    save_t3,
)

DATA = Path("data")


def main() -> None:
    gold = load_gold(DATA / "train.csv")
    gold.to_csv(DATA / "holdout_gold.csv", index=False)
    print(f"T2 gold: {len(gold)} rows -> data/holdout_gold.csv")

    pool = load_pool(
        [
            DATA / "pools/radek1/6000_train_examples.csv",
            DATA / "pools/radek1/extra_train_set.csv",
        ]
    )
    print(f"raw pool: {len(pool)} rows")
    dup_mask = near_duplicate_mask(pool["prompt"], gold["prompt"])
    print(f"near-duplicates of gold found in pool: {int(dup_mask.sum())}")

    t1 = build_t1(pool, gold, n=1500, seed=42)
    t1.to_csv(DATA / "t1_dev.csv", index=False)
    print(f"T1 dev: {len(t1)} rows -> data/t1_dev.csv")

    arc_raw = pd.read_parquet(DATA / "ood_raw/arc_challenge_test.parquet")
    mmlu_raw = pd.read_parquet(DATA / "ood_raw/mmlu_all_test.parquet")
    arc_norm = normalize_arc(arc_raw)
    mmlu_norm = normalize_mmlu(mmlu_raw, subjects=MMLU_STEM_SUBJECTS)
    print(f"ARC-Challenge normalized: {len(arc_norm)} rows")
    print(f"MMLU-STEM normalized: {len(mmlu_norm)} rows across {mmlu_norm['subject'].nunique()} subjects")

    t3 = build_t3(arc_norm, mmlu_norm, n_each=500, seed=42)
    save_t3(t3, DATA / "t3_ood.parquet")
    print(f"T3 OOD: {len(t3)} rows -> data/t3_ood.parquet")


if __name__ == "__main__":
    main()
