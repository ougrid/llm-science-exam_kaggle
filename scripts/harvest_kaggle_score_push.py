"""Harvest the corrected-LR Kaggle runs and log them honestly.

Downloads each finished kernel's outputs, parses result_summary.txt, prints the
headline numbers next to the reference points they must be read against, and
appends a row per kernel to experiments/log.csv.

Deliberately does NOT compute a paired bootstrap against the pre-fix runs: the
pre-fix checkpoints were selected by an untrained model's noise (and the row-7
checkpoint was cached in /tmp, which a machine restart wiped today), so a
paired comparison against them would be comparing a trained model to a
coin flip and would overstate precision. The honest framing is a fresh
absolute number with its CI, against the analytic baseline.

Usage from the repo root:
  python scripts/harvest_kaggle_score_push.py
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pandas as pd

from llmsci.experiment import git_sha, log_experiment
from llmsci.metrics import random_baseline_map_at_k

KERNELS = {
    "ougridd/day3-score-push-base": {
        "label": "deberta-v3-base",
        "lr": "2e-5",
        "steps": "~2452 (2 epochs, batch4 x accum8)",
    },
    "ougridd/day3-score-push-large": {
        "label": "deberta-v3-large",
        "lr": "1e-5",
        "steps": "~1226 (1 epoch, batch2 x accum16)",
    },
}
OUT_ROOT = Path("data/kaggle_score_push")
BASELINE = random_baseline_map_at_k()


def kernel_status(slug: str) -> str:
    r = subprocess.run(
        ["kaggle", "kernels", "status", slug], capture_output=True, text=True, check=False
    )
    return (r.stdout + r.stderr).strip()


def fetch(slug: str, dest: Path) -> bool:
    dest.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(
        ["kaggle", "kernels", "output", slug, "-p", str(dest)],
        capture_output=True, text=True, check=False,
    )
    if r.returncode != 0:
        print(f"  fetch failed: {(r.stdout + r.stderr).strip()[:300]}")
        return False
    return (dest / "result_summary.txt").exists()


def parse_summary(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            out[k.strip()] = v.strip()
    return out


def parse_map3(field: str) -> tuple[float, float, float] | None:
    m = re.match(r"([\d.]+)\s*\[([\d.]+),([\d.]+)\]", field or "")
    return (float(m.group(1)), float(m.group(2)), float(m.group(3))) if m else None


def main() -> None:
    for slug, meta in KERNELS.items():
        name = slug.split("/")[-1]
        print(f"=== {slug} ({meta['label']}, lr={meta['lr']}) ===")
        print(f"  status: {kernel_status(slug)}")
        dest = OUT_ROOT / name
        if not fetch(slug, dest):
            print("  no result_summary.txt yet -- kernel likely still running; rerun later\n")
            continue

        summary = parse_summary(dest / "result_summary.txt")
        best = parse_map3(summary.get("best_map3", ""))
        final = parse_map3(summary.get("final_map3", ""))
        print(f"  config: {summary.get('config', '?')}")
        print(f"  best_optim_step: {summary.get('best_optim_step', '?')}")
        if best:
            resolved = best[1] > BASELINE
            print(f"  BEST full-T1 MAP@3: {best[0]:.4f} [{best[1]:.4f},{best[2]:.4f}]  "
                  f"(baseline {BASELINE:.4f}; CI clears baseline: {resolved})")
        if final:
            print(f"  end-of-training full-T1 MAP@3: {final[0]:.4f} [{final[1]:.4f},{final[2]:.4f}]")
        print(f"  stopped_early_on_time_budget: {summary.get('stopped_early_on_time_budget', '?')}")
        print(f"  train_seconds: {summary.get('train_seconds', '?')}")

        if best:
            log_experiment({
                "date": pd.Timestamp.now().isoformat(timespec="seconds"),
                "git_sha": git_sha(),
                "config": (
                    f"{meta['label']}_open-book_GENERAL-CORPUS-BIGPOOL_CORRECTED-LR{meta['lr']}"
                    f"_n39249_KAGGLE-TRAINED-AND-SCORED"
                ),
                "tier": "T1+own-BM25-general-corpus-context",
                "n": 1500,
                "map3_mean": round(best[0], 4),
                "map3_ci_lower": round(best[1], 4),
                "map3_ci_upper": round(best[2], 4),
                "random_baseline": round(BASELINE, 4),
                "train_seconds": summary.get("train_seconds", ""),
                "eval_seconds": "",
                "hypothesis": (
                    "does correcting the optimization (lr 5e-6 -> "
                    f"{meta['lr']}, 4,978 -> 39,249 rows, {meta['steps']}) lift the reader off the "
                    "ln(5) uniform-prediction floor it sat at in every previous run, and how far"
                ),
                "notes": (
                    f"First run in this project trained with a corrected LR after "
                    f"scripts/diagnose_overfit_sanity.py showed every prior run's loss was pinned at "
                    f"ln(5)=1.6094 (uniform over 5 options) -- i.e. the reader had never trained. "
                    f"Model {meta['label']}, lr={meta['lr']}, {meta['steps']}, general-Wikipedia "
                    f"own-retrieval context. Checkpoint SELECTED on a fixed 500-row T1 subset (CI "
                    f"half-width ~±0.032) then RE-SCORED on the full 1,500 rows, which is the number "
                    f"reported here. End-of-training state: "
                    f"{final[0]:.4f} [{final[1]:.4f},{final[2]:.4f}]. " if final else ""
                ) + (
                    f"stopped_early_on_time_budget={summary.get('stopped_early_on_time_budget', '?')} "
                    f"(TIME_BUDGET_S sized so the kernel FINISHES -- Kaggle only serves output files "
                    f"from finished runs). NOT paired-bootstrapped against the pre-fix runs on "
                    f"purpose: those checkpoints were selected by an untrained model's noise, so a "
                    f"paired delta against them would overstate precision. CAVEAT: trained AND "
                    f"scored on Kaggle, and this project has a documented Kaggle-vs-local scoring "
                    f"discrepancy (see the CRITICAL_diagnostic_kaggle_vs_local row) -- comparable to "
                    f"other Kaggle-scored numbers, not directly to locally-scored ones."
                ),
            })
            print("  -> logged to experiments/log.csv")
        print()


if __name__ == "__main__":
    main()
