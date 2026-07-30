"""Paired bootstrap between two runs of this track on the same 1,500 T1 rows.

CLAUDE.md: "Comparisons use paired bootstrap, not two unpaired CIs." Both runs
score the identical, row-aligned T1 frame, so per-row AP@3 differences are
paired and `llmsci.metrics.paired_bootstrap` applies directly. Also reports
`minimum_detectable_effect`, so a null result can be labelled "not resolved by
my eval" rather than silently read as "no effect."

Usage: python compare.py faithful reranked
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from llmsci.metrics import (
    bootstrap_ci,
    minimum_detectable_effect,
    paired_bootstrap,
    random_baseline_map_at_k,
)

RESULTS = Path(__file__).resolve().parent.parent / "results"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("baseline")
    p.add_argument("variant")
    args = p.parse_args()

    a = np.load(RESULTS / f"t1_ap_{args.baseline}.npy")
    b = np.load(RESULTS / f"t1_ap_{args.variant}.npy")
    if len(a) != len(b):
        raise ValueError(f"row counts differ: {len(a)} vs {len(b)}")

    am, alo, ahi = bootstrap_ci(a, n_resamples=10_000, seed=0)
    bm, blo, bhi = bootstrap_ci(b, n_resamples=10_000, seed=0)
    dm, dlo, dhi = paired_bootstrap(a, b, n_resamples=10_000, seed=0)
    mde = minimum_detectable_effect(a, b)
    flipped = float((a != b).mean())
    resolved = (dlo > 0) or (dhi < 0)

    print(f"n = {len(a)} T1 dev rows, random baseline {random_baseline_map_at_k():.4f}\n")
    print(f"{args.baseline:<12} MAP@3 {am:.4f}  95% CI [{alo:.4f}, {ahi:.4f}]")
    print(f"{args.variant:<12} MAP@3 {bm:.4f}  95% CI [{blo:.4f}, {bhi:.4f}]")
    print(f"\npaired delta ({args.variant} - {args.baseline}): {dm:+.4f}  "
          f"95% CI [{dlo:+.4f}, {dhi:+.4f}]")
    print(f"rows whose AP@3 changed: {flipped:.1%}")
    print(f"minimum detectable effect at n={len(a)}: +/-{mde:.4f}")
    print(f"\nverdict: {'RESOLVED' if resolved else 'NOT RESOLVED BY THIS EVAL (CI includes 0)'}")

    with open(RESULTS / f"compare_{args.baseline}_vs_{args.variant}.json", "w") as f:
        json.dump({
            "n": len(a),
            "baseline": args.baseline, "variant": args.variant,
            "baseline_map3": am, "baseline_ci": [alo, ahi],
            "variant_map3": bm, "variant_ci": [blo, bhi],
            "paired_delta": dm, "paired_delta_ci": [dlo, dhi],
            "rows_changed": flipped,
            "mde": mde,
            "resolved": bool(resolved),
        }, f, indent=2)


if __name__ == "__main__":
    main()
