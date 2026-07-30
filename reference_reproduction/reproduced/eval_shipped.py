"""Score a checkpoint on T1 under part 2's ACTUAL inference format.

Part 2 does not reuse part 1's tokenization. It sets

    prompt <- context[:1750] + " #### " + prompt

then tokenizes `tokenizer(prompt_field, option, truncation=True)` with
add_special_tokens at its default and NO max_length. The token layout still
comes out as `[CLS] ctx #### prompt [SEP] option [SEP]`, but the length budget
does not match training's 256 -- the 1750-character clip (~400 tokens) is the
real limiter. See ../NOTES.md.

Two reasons to implement this separately from common.OpenBookMCDataset:

1. `mgoksu/llm-science-run-context-2` is the second leg of part 2's 50/50
   ensemble and the checkpoint the published 0.823761 actually ran. It is a
   2023 artifact trained on far more data than the 1,024 rows used here, so
   its T1 score is a reference point that does not depend on my training at
   all -- arguably the most useful single number in this track.
2. Scoring the same weights under both formats measures what the notebook's
   own train/inference length mismatch costs, which is this project's own
   Day-2 theme applied to someone else's pipeline.

Usage:
    python eval_shipped.py --model-dir ../models/mgoksu-run-context-2 --tag mgoksu
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForMultipleChoice, AutoTokenizer

from common import OPTIONS, MCCollator, logits_to_ranked_labels
from llmsci.gpu_guard import cap_memory_fraction
from llmsci.metrics import average_precision_scores, bootstrap_ci, random_baseline_map_at_k

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
RESULTS = HERE.parent / "results"

CONTEXT_CHAR_CLIP = 1750  # part 2, cell 27


class Part2InferenceDataset(Dataset):
    """Part 2's inference tokenization, including its length behaviour."""

    def __init__(self, df, tokenizer, context_col: str = "context",
                 char_clip: int = CONTEXT_CHAR_CLIP, max_length: int | None = None):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.context_col = context_col
        self.char_clip = char_clip
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        ctx = str(row[self.context_col])[: self.char_clip]
        first = f"{ctx} #### {row['prompt']}"
        kwargs = {"truncation": True}
        if self.max_length is not None:
            kwargs["max_length"] = self.max_length
        enc = self.tokenizer([first] * 5, [str(row[c]) for c in OPTIONS], **kwargs)
        item = dict(enc)
        if row["answer"] in OPTIONS:
            item["label"] = OPTIONS.index(row["answer"])
        return item


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--context-col", default="context")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--max-length", type=int, default=None,
                   help="omit to reproduce part 2 (no max_length); set to cap")
    args = p.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        cap_memory_fraction(0.975)

    df = pd.read_parquet(DATA / "t1_eval_cdeotte_ctx.parquet")
    df[args.context_col] = df[args.context_col].fillna("").astype(str)

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForMultipleChoice.from_pretrained(args.model_dir).to(device)
    model.eval()
    print(f"[eval_shipped] {args.model_dir} tag={args.tag} ctx={args.context_col} "
          f"clip={CONTEXT_CHAR_CLIP} chars, max_length={args.max_length}")

    ds = Part2InferenceDataset(df, tokenizer, args.context_col, max_length=args.max_length)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        collate_fn=MCCollator(tokenizer))

    lens, out, start = [], [], time.time()
    with torch.no_grad():
        for i, batch in enumerate(loader):
            batch.pop("labels", None)
            lens.append(batch["input_ids"].shape[-1])
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out.append(model(**batch).logits.float().cpu().numpy())
            if (i + 1) % 100 == 0 or (i + 1) == len(loader):
                el = time.time() - start
                rate = (i + 1) / el
                print(f"  batch {i+1}/{len(loader)} elapsed {el:.0f}s rate {rate:.1f} b/s "
                      f"eta {(len(loader)-i-1)/rate:.0f}s", flush=True)
    logits = np.concatenate(out, axis=0)

    ap = average_precision_scores(df["answer"].tolist(), logits_to_ranked_labels(logits, 3), k=3)
    mean, lo, hi = bootstrap_ci(ap, n_resamples=10_000, seed=0)
    lens = np.array(lens)
    print(f"\nsequence length actually fed: mean {lens.mean():.0f} median {np.median(lens):.0f} "
          f"max {lens.max()} (training used 256)")
    print(f"T1 MAP@3 (n={len(df)}): {mean:.4f}  95% CI [{lo:.4f}, {hi:.4f}]  "
          f"(random {random_baseline_map_at_k():.4f})")

    np.save(RESULTS / f"t1_logits_{args.tag}.npy", logits)
    np.save(RESULTS / f"t1_ap_{args.tag}.npy", ap)
    with open(RESULTS / f"summary_{args.tag}.json", "w") as f:
        json.dump({
            "tag": args.tag, "model_dir": args.model_dir,
            "format": "part2_inference", "context_char_clip": CONTEXT_CHAR_CLIP,
            "max_length": args.max_length,
            "seq_len_mean": float(lens.mean()), "seq_len_max": int(lens.max()),
            "n_t1": len(df), "t1_map3": mean, "t1_ci_lower": lo, "t1_ci_upper": hi,
            "random_baseline": random_baseline_map_at_k(),
        }, f, indent=2)
    print(f"wrote results for tag={args.tag}")


if __name__ == "__main__":
    main()
