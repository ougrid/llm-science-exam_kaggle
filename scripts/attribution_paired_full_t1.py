"""The attribution table, done properly: same rows, same k, paired bootstrap.

Why this replaces the old table. The README reported a "reader-attributable loss"
of 0.4130 as ceiling (0.7970) minus our reader (0.3840). Two things were wrong
with it:

  1. It was quantifying BUGS, not a training gap. Our 0.3840 reader was training
     fp16 *parameters* at an inherited learning rate that measurement later showed
     cannot learn (see DEVLOG "the fifth hypothesis" and "hypothesis six").
  2. The two numbers came from DIFFERENT ROW SAMPLES. The 0.7970 ceiling was a
     500-row sample of T1 at random_state=0; our reader's figure was on other
     rows. That is not a valid subtraction, and the size of the error it can hide
     was just measured directly: a 500-row seed-42 sample of this same T1 scored
     0.0321 lower than its own 1,000-row complement. Sample choice moves the
     answer by about one CI half-width, which is the same order as some of the
     effects being claimed.

So: both readers, the SAME full 1,500 rows, per-row AP@3, and a paired bootstrap
on the difference rather than two unpaired CIs (CLAUDE.md -- two configs agree on
most rows, so the SD of the difference is far smaller than either alone).

What the output means:
  * `paired delta` is the honest reader-attributable loss. Everything ABOVE the
    ceiling is what better retrieval could add; everything BELOW the delta is what
    better reader training could add.
  * `disagreement rate` is how often the two readers differ at all. A paired test's
    resolution depends on it, so it is printed rather than assumed.

Run: python scripts/attribution_paired_full_t1.py
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
from llmsci.reader.mc import DataCollatorForMultipleChoice, logits_to_ranked_labels

DATA = Path("data")
T1_CTX = DATA / "t1_dev_own_context_general_big.parquet"
PUBLIC = Path("reference_reproduction/models/mgoksu-run-context-2")
OURS = DATA / "checkpoints" / "base-fixed-lr1e-4"
OPTIONS = ["A", "B", "C", "D", "E"]
BASELINE = random_baseline_map_at_k()
MAX_CONTEXT_CHARS = 8_000
EVAL_BATCH = 2

# The two readers were trained with DIFFERENT input layouts, and each must be fed
# the one it was trained on -- otherwise this measures a format mismatch rather
# than reader quality. Verified equivalent earlier: cell C of
# diagnose_train_data_and_format.py showed the public reader loses essentially
# nothing (0.7947 vs 0.7970) when read in our layout, so the layouts are
# interchangeable for it and using each reader's native one is the fair choice.
LAYOUT_PUBLIC = "ctx_hash_prompt"    # "{context} #### {prompt}" / option, maxlen 512
LAYOUT_OURS = "ctx_then_prompt_opt"  # context / "{prompt} {option}", maxlen 384


def encode(tok, row, layout):
    if layout == LAYOUT_PUBLIC:
        first = [f"{str(row['context'])[:1750]} #### {row['prompt']}"] * 5
        second = [str(row[c]) for c in OPTIONS]
        maxlen = 512
    else:
        first = [str(row["context"])[:MAX_CONTEXT_CHARS]] * 5
        second = [f"{row['prompt']} {row[c]}" for c in OPTIONS]
        maxlen = 384
    return dict(tok(first, second, truncation="only_first", max_length=maxlen))


def per_row_ap(ckpt: Path, df, layout: str, device) -> np.ndarray:
    from torch.utils.data import Dataset
    from transformers import AutoModelForMultipleChoice, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(ckpt)
    model = AutoModelForMultipleChoice.from_pretrained(ckpt, dtype=torch.float32).to(device).eval()

    class DS(Dataset):
        def __len__(self):
            return len(df)

        def __getitem__(self, i):
            return encode(tok, df.iloc[i], layout)

    loader = DataLoader(DS(), batch_size=EVAL_BATCH, shuffle=False,
                        collate_fn=DataCollatorForMultipleChoice(tok))
    out = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16,
                                         enabled=(device.type == "cuda")):
        for i, b in enumerate(loader):
            b.pop("labels", None)
            out.append(model(**{k: v.to(device) for k, v in b.items()}).logits.float().cpu().numpy())
            if (i + 1) % 150 == 0:
                print(f"      {ckpt.name}: batch {i+1}/{len(loader)}", flush=True)
    logits = np.concatenate(out, axis=0)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return average_precision_scores(df["answer"].tolist(), logits_to_ranked_labels(logits, k=3), k=3)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", default=str(OURS))
    args = ap.parse_args()
    ours = Path(args.ours)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        cap_memory_fraction(0.975)
    df = pd.read_parquet(T1_CTX)
    df["context"] = df["context"].str.slice(0, MAX_CONTEXT_CHARS)
    print(f"eval: ALL {len(df)} rows of T1, identical for both readers "
          f"| baseline {BASELINE:.4f}")

    if not PUBLIC.exists() or not ours.exists():
        print(f"missing checkpoint: public={PUBLIC.exists()} ours={ours.exists()}")
        return

    s_pub = per_row_ap(PUBLIC, df, LAYOUT_PUBLIC, device)
    m_pub = bootstrap_ci(s_pub, n_resamples=10_000, seed=0)
    print(f"  known-good public reader : MAP@3 {m_pub[0]:.4f} [{m_pub[1]:.4f},{m_pub[2]:.4f}]")

    s_our = per_row_ap(ours, df, LAYOUT_OURS, device)
    m_our = bootstrap_ci(s_our, n_resamples=10_000, seed=0)
    print(f"  our trained reader       : MAP@3 {m_our[0]:.4f} [{m_our[1]:.4f},{m_our[2]:.4f}]")

    d, dlo, dhi = paired_bootstrap(s_our, s_pub, n_resamples=10_000, seed=0)  # mean(pub - ours)
    disagree = float((s_our != s_pub).mean())
    print(f"\n  READER-ATTRIBUTABLE LOSS (paired): {d:+.4f} [{dlo:+.4f},{dhi:+.4f}]")
    print(f"  disagreement rate: {disagree:.3f} of rows")
    print(f"  resolved: {'yes -- CI excludes 0' if (dlo > 0 or dhi < 0) else 'NO -- CI includes 0'}")
    print(f"\n  previously CLAIMED in README: 0.4130 -- which was quantifying two bugs")
    print(f"  (fp16 parameters + an inherited lr) AND subtracting across different")
    print(f"  row samples. Corrected value above.")
    if d < 0.4130:
        print(f"  => {(1 - d/0.4130)*100:.0f}% of the old 0.4130 was bugs, not a training gap.")

    Path("reports").mkdir(exist_ok=True)
    Path("reports/attribution_paired.json").write_text(json.dumps({
        "eval_rows": len(df), "baseline": BASELINE,
        "public_reader": list(m_pub), "our_reader": list(m_our),
        "reader_attributable_loss_paired": [d, dlo, dhi],
        "disagreement_rate": disagree,
        "resolved": bool(dlo > 0 or dhi < 0),
        "superseded_claim": 0.4130,
        "our_checkpoint": str(ours),
    }, indent=2))
    print("wrote reports/attribution_paired.json")


if __name__ == "__main__":
    main()
