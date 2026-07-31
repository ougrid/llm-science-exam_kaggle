"""Did 8.6x more training data help? Paired bootstrap, on the same frozen rows.

The question. We trained on 4,586 of 39,249 available rows because of hypothesis 4
(a train/eval generator mismatch), which cell B of
diagnose_train_data_and_format.py falsified -- a known-good reader scores 0.8037
[0.7743,0.8320] on the TRAIN file versus 0.7970 [0.7687,0.8247] on the eval file,
so the two are equally readable. With the reason gone, the restriction cost 88% of
the pool for nothing. This measures what that was worth.

Why paired and not two CIs. Both checkpoints are scored on the SAME frozen
1,500-row T1, so the correct test is a bootstrap on the per-row AP@3 DIFFERENCE.
Two similar configs agree on most rows, so the SD of the difference is far smaller
than either model's own SD -- roughly a 5x resolution gain for free (CLAUDE.md).
Comparing two unpaired CIs here would be the wrong test and would likely call a
real effect "unresolved".

Thresholds, declared before the numbers exist (see also the same list in
DEVLOG.md, written before this ran):

  * paired CI excludes 0 and delta > +0.03  -> data quantity was the binding
    constraint, and the hypothesis-4 restriction cost real score on top of the
    debugging time it consumed.
  * paired CI includes 0                    -> capacity-bound at 22.2M trainable
    parameters; the next lever is the reader, not the data. Report "not resolved
    by my eval" -- do NOT round up.
  * delta negative and CI excludes 0        -> the 7 extra generator sources hurt,
    source matching mattered after all, and cell B needs re-examining. This is the
    outcome I least expect and would most need to report honestly, since it
    partially rehabilitates a hypothesis I spent hours declaring dead.

Run: python scripts/compare_pool_size_paired.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from llmsci.gpu_guard import cap_memory_fraction
from llmsci.metrics import (
    average_precision_scores,
    bootstrap_ci,
    paired_bootstrap,
    random_baseline_map_at_k,
)
from llmsci.reader.mc import (
    DataCollatorForMultipleChoice,
    MultipleChoiceDataset,
    logits_to_ranked_labels,
)

DATA = Path("data")
T1_CTX = DATA / "t1_dev_own_context_general_big.parquet"
CKPT_SMALL = DATA / "checkpoints" / "base-fixed-lr1e-4"
CKPT_BIG = DATA / "checkpoints" / "base-fixed-lr1e-4-bigpool"

MAX_LENGTH = 384
MAX_CONTEXT_CHARS = 8_000
EVAL_BATCH = 4
BASELINE = random_baseline_map_at_k()
CEILING = 0.7970  # known-good public reader, same eval set, same retrieval


def per_row_ap(ckpt: Path, df, device) -> np.ndarray:
    from transformers import AutoModelForMultipleChoice, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(ckpt)
    model = AutoModelForMultipleChoice.from_pretrained(ckpt, dtype=torch.float32).to(device).eval()
    loader = DataLoader(
        MultipleChoiceDataset(df, tok, max_length=MAX_LENGTH, context_col="context"),
        batch_size=EVAL_BATCH, shuffle=False, collate_fn=DataCollatorForMultipleChoice(tok),
    )
    out = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16,
                                         enabled=(device.type == "cuda")):
        for b in loader:
            b.pop("labels", None)
            out.append(model(**{k: v.to(device) for k, v in b.items()}).logits.float().cpu().numpy())
    logits = np.concatenate(out, axis=0)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return average_precision_scores(df["answer"].tolist(), logits_to_ranked_labels(logits, k=3), k=3)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--small", default=str(CKPT_SMALL))
    ap.add_argument("--big", default=str(CKPT_BIG))
    args = ap.parse_args()
    small, big = Path(args.small), Path(args.big)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        cap_memory_fraction(0.975)
    df = pd.read_parquet(T1_CTX)
    df["context"] = df["context"].str.slice(0, MAX_CONTEXT_CHARS)
    print(f"frozen eval set: {len(df)} rows | baseline {BASELINE:.4f} | ceiling {CEILING:.4f}")

    rows = []
    for tag, ckpt in (("4,586-row pool", small), ("39,249-row pool", big)):
        if not ckpt.exists():
            print(f"  {tag}: {ckpt} MISSING -- run it first")
            return
        prog = ckpt / "result_summary.json"
        n = json.loads(prog.read_text()).get("train_rows") if prog.exists() else "?"
        s = per_row_ap(ckpt, df, device)
        m, lo, hi = bootstrap_ci(s, n_resamples=10_000, seed=0)
        print(f"  {tag:16s} (train_rows={n})  MAP@3 {m:.4f} [{lo:.4f},{hi:.4f}]", flush=True)
        rows.append((tag, s, (m, lo, hi)))

    (_, s_small, r_small), (_, s_big, r_big) = rows
    d, dlo, dhi = paired_bootstrap(s_small, s_big, n_resamples=10_000, seed=0)
    # paired_bootstrap(a, b) returns mean(b - a), so this is big MINUS small.
    resolved = dlo > 0 or dhi < 0
    print(f"\n  paired delta (39,249 pool - 4,586 pool): {d:+.4f} [{dlo:+.4f},{dhi:+.4f}]")
    print(f"  disagreement rate: {float((s_small != s_big).mean()):.3f} of rows")
    print("\nREADING (thresholds were declared in the module docstring before this ran)")
    if resolved and d > 0.03:
        print(f"  Data quantity WAS the binding constraint (+{d:.4f}, CI excludes 0).")
        print("  The hypothesis-4 row restriction cost real score, not just debugging time.")
    elif resolved and d < 0:
        print(f"  MORE DATA HURT ({d:+.4f}, CI excludes 0). Source matching mattered after all;")
        print("  cell B of diagnose_train_data_and_format.py needs re-examining. This")
        print("  partially rehabilitates hypothesis 4.")
    elif resolved:
        print(f"  Resolved but small (+{d:.4f}). Real, below the 0.03 threshold I set for")
        print("  calling data quantity the binding constraint.")
    else:
        print(f"  NOT RESOLVED by this eval ({d:+.4f}, CI includes 0). Capacity-bound at")
        print("  22.2M trainable parameters is the live explanation; the next lever is the")
        print("  reader, not the data. Reported as unresolved -- not rounded up.")
    gap = CEILING - max(r_small[0], r_big[0])
    print(f"\n  best own model {max(r_small[0], r_big[0]):.4f} vs ceiling {CEILING:.4f} "
          f"-> {gap:.4f} still attributable to the reader")


if __name__ == "__main__":
    main()
