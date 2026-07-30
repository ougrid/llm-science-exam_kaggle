"""Shared helpers for recording runs to experiments/log.csv.

Every training script logs through here so the schema stays identical across
closed-book, open-book, and later ablations -- `experiments/log.csv` is only
useful as a comparison table if every row has the same columns.
"""

from __future__ import annotations

import csv
import subprocess
from pathlib import Path

LOG_FIELDS = [
    "date",
    "git_sha",
    "config",
    "tier",
    "n",
    "map3_mean",
    "map3_ci_lower",
    "map3_ci_upper",
    "random_baseline",
    "train_seconds",
    "eval_seconds",
    "hypothesis",
    "notes",
]


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


def log_experiment(row: dict, log_path: Path = Path("experiments/log.csv")) -> None:
    log_path.parent.mkdir(exist_ok=True)
    is_new = not log_path.exists()
    row = {**{k: "" for k in LOG_FIELDS}, **row}  # ensure every field is present
    with open(log_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)
