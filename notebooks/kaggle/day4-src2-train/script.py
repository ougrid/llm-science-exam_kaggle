"""DAY 4 (rerun): the FIRST run in this project whose optimizer can actually work.

WHY THE PREVIOUS FOUR TRAINING RUNS FAILED -- and it was not any of the four
things I claimed. Every run pinned train_loss at ln(5)=1.6094 (measured:
1.6118..1.6173 across 1,000 optimizer steps and 4 epochs of 4,586 rows with
77.2M trainable parameters). The cause was one library default:

  transformers 5.x makes from_pretrained follow the CHECKPOINT's stored dtype,
  and both deberta-v3-base and -large ship fp16. So the bare call returned fp16
  PARAMETERS -- not fp16 compute with fp32 master weights (that is mixed
  precision), but genuinely half-precision weights that AdamW updated in place.

An AdamW step is ~lr in magnitude, so the only question is whether lr is
REPRESENTABLE as a change to the weight. Measured, 20 steps at lr=2e-5 with a
constant-sign gradient -- the easiest possible case:

    w = 0.03  ->  fp16 moves 3.05e-04 vs fp32's 4.00e-04   (76% of intended)
    w >= 0.1  ->  fp16 moves 0.00e+00                      (FROZEN, forever)

fp16 ULP grows with magnitude, so the larger a weight is the smaller lr looks
beside it. deberta's LayerNorm weights sit near 1.0, so most of the network was
frozen solid; only the small head weights moved, in erratic ULP jumps. On this
exact recipe, one step moved classifier.weight by 1.072e-01 in fp16 against a
correct 2.001e-05 in fp32. Controlled test, 16 rows / 60 steps / one variable:

    fp16: loss 1.53 -> 4.32 -> 1.5687   parked at ln(5)
    fp32: loss 1.68 -> 0.47 -> 0.1013   learns

WHAT WAS RULED OUT FIRST, so this is a diagnosis and not a fifth guess.
Using the known-good public reader as an INSTRUMENT to test our data
(scripts/diagnose_train_data_and_format.py), three cells one variable apart:

    A  eval file  + reference format (control)   0.7970 [0.7687, 0.8247]
    B  TRAIN file + reference format             0.8037 [0.7743, 0.8320]
    C  eval file  + THIS script's format         0.7947 [0.7657, 0.8240]

All three overlap. So the training pool's labels are correct, its context
supports its answers, and this script's input format costs essentially nothing.
Cell B also kills the story the previous version of this file was built on --
a train/eval generator mismatch -- because train and eval turn out to be equally
readable by the same instrument. Gradients were always healthy too (frozen
groups grad=None, classifier 3.83, pooler 4.26, layers 18-23 in 0.08-0.41).

WHAT THIS RUN IS. The same source-matched 4,586 rows with our own general-corpus
BM25 context (so train and eval retrieval match by construction, per CLAUDE.md)
and the same frozen-layer recipe -- but with fp32 MASTER WEIGHTS and fp16
autocast COMPUTE. Both halves are load-bearing: fp16 weights make the optimizer
a no-op, and fp32 compute would roughly double activation memory and risk OOM at
batch 2 x 384 on a 16 GB T4.

WHAT TO READ IN THE LOG. train_loss must FALL BELOW 1.6094. If it sits there
again, the fix did not take and nothing else in the log matters.

REFERENCE POINTS. Random baseline 0.3667. Our previous best own-model 0.6086
(row 6) and the 0.3840/0.3746 nulls are all products of the bug above and are
floors, not targets. The honest ceiling is 0.7970 -- the known-good reader on
this same eval set with this same retrieval.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForMultipleChoice,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    get_linear_schedule_with_warmup,
)

OUT = Path("/kaggle/working")
OPTION_COLUMNS = ["A", "B", "C", "D", "E"]

MODEL_NAME = "microsoft/deberta-v3-large"
MAX_LENGTH = 384
MAX_CONTEXT_CHARS = 8_000
BATCH_SIZE = 2  # batch 4 OOM'd on T4 in disentangled attention; 2 is measured-safe
GRAD_ACCUM_STEPS = 8  # effective batch 16, matching cdeotte part 2
EVAL_BATCH_SIZE = 8
# Sized from the MEASURED rate of the previous run: 6.4 s per optimizer step
# (80 sequences of 384 tokens through deberta-v3-large on a T4). 6 epochs =
# 1,716 steps ~= 3.0 h, leaving room inside TIME_BUDGET_S for two full 1,500-row
# evals at the end. The point is for the linear LR schedule to actually COMPLETE:
# a run guillotined by the time budget at 34% LR never anneals, and the previous
# 12-epoch plan could not finish.
EPOCHS = 6
EVAL_EVERY_STEPS = 100
LR = 2e-5  # cdeotte part 2's value, paired with the freezing below
SEED = 42
# Sized to FINISH, not to train as long as possible: a kernel still running when
# you go to collect it yields nothing, because `kaggle kernels output` only
# serves files from finished runs. 6 epochs at the measured 6.4 s/step is ~3.0 h,
# so this budget is slack, not a guillotine -- and the graceful stop below always
# writes the best checkpoint plus result_summary.txt regardless.
TIME_BUDGET_S = 4 * 3600  # session cap is 6h
# In-training evals score a fixed 500-row T1 subset, not all 1,500: a full eval
# costs ~2.6 min (base) / ~5.2 min (large), so evaluating in full every 100
# steps would spend more wall clock on evaluation than on training. The subset
# is only for checkpoint SELECTION (its 95% CI half-width is ~±0.032 vs ±0.018
# on the full set -- noisier, accepted deliberately); the reported number at
# the end is always a full 1,500-row eval.
EVAL_SUBSET_N = 500
FREEZE_EMBEDDINGS = True
N_FROZEN_LAYERS = 18  # of 24, per cdeotte part 2
RANDOM_LOSS = math.log(5)


def find_data_file(filename: str) -> Path:
    matches = glob(f"/kaggle/input/**/{filename}", recursive=True)
    if not matches:
        print("DEBUG: /kaggle/input contents:")
        for p in glob("/kaggle/input/**/*", recursive=True):
            print(" ", p)
        raise FileNotFoundError(f"{filename} not found anywhere under /kaggle/input")
    return Path(matches[0])


def average_precision_at_k(actual: str, predicted: list[str], k: int = 3) -> float:
    for i, p in enumerate(predicted[:k]):
        if p == actual:
            return 1.0 / (i + 1)
    return 0.0


def average_precision_scores(y_true: list[str], y_pred: list[list[str]], k: int = 3) -> np.ndarray:
    return np.array([average_precision_at_k(a, p, k) for a, p in zip(y_true, y_pred)])


def bootstrap_ci(scores: np.ndarray, n_resamples: int = 10_000, ci: float = 0.95, seed: int = 0):
    rng = np.random.default_rng(seed)
    scores = np.asarray(scores)
    n = len(scores)
    means = np.empty(n_resamples)
    for i in range(n_resamples):
        means[i] = scores[rng.integers(0, n, n)].mean()
    lo = (1 - ci) / 2
    return float(scores.mean()), float(np.quantile(means, lo)), float(np.quantile(means, 1 - lo))


def random_baseline_map_at_k(num_options: int = 5, k: int = 3) -> float:
    return sum(1.0 / i for i in range(1, k + 1)) / num_options


def build_choice_texts(prompt: str, options: list[str], context: str = ""):
    return [context] * len(options), [f"{prompt} {opt}" for opt in options]


class MultipleChoiceDataset(Dataset):
    def __init__(self, df, tokenizer: PreTrainedTokenizerBase, max_length: int = 256, context_col: str | None = None):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.context_col = context_col

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx]
        options = [row[c] for c in OPTION_COLUMNS]
        context = row[self.context_col] if self.context_col else ""
        first, second = build_choice_texts(row["prompt"], options, context)
        encoded = self.tokenizer(first, second, truncation="only_first", max_length=self.max_length)
        item = dict(encoded)
        if "answer" in row:
            item["label"] = OPTION_COLUMNS.index(row["answer"])
        return item


@dataclass
class DataCollatorForMultipleChoice:
    tokenizer: PreTrainedTokenizerBase
    padding: bool | str = True

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        has_labels = "label" in features[0]
        labels = [f.pop("label") for f in features] if has_labels else None
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])
        flattened = [{k: v[i] for k, v in f.items()} for f in features for i in range(num_choices)]
        batch = self.tokenizer.pad(flattened, padding=self.padding, return_tensors="pt")
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        if has_labels:
            batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch


def logits_to_ranked_labels(logits: np.ndarray, k: int = 3) -> list[list[str]]:
    order = np.argsort(-logits, axis=1)
    return [[OPTION_COLUMNS[i] for i in row[:k]] for row in order]


def evaluate(model, tokenizer, df, collator, device):
    model.eval()
    # Eval under autocast too, so the scored numbers come from the same numeric
    # path as training and the reported MAP@3 matches what the checkpoint does.
    ds = MultipleChoiceDataset(df, tokenizer, max_length=MAX_LENGTH, context_col="context")
    loader = DataLoader(ds, batch_size=EVAL_BATCH_SIZE, shuffle=False, collate_fn=collator)
    all_logits = []
    amp = torch.autocast("cuda", dtype=torch.float16) if device.type == "cuda" \
        else torch.autocast("cpu", enabled=False)
    with torch.no_grad(), amp:
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            all_logits.append(model(**batch).logits.float().cpu().numpy())
    logits = np.concatenate(all_logits, axis=0)
    scores = average_precision_scores(df["answer"].tolist(), logits_to_ranked_labels(logits, k=3), k=3)
    model.train()
    return (*bootstrap_ci(scores), logits)


def write_summary(path, best, final, best_step, n_train, train_seconds, stopped_early):
    with open(path, "w") as f:
        f.write(
            f"config: {MODEL_NAME}_SOURCE-MATCHED-src2_lr{LR}_bs{BATCH_SIZE}x{GRAD_ACCUM_STEPS}_"
            f"{EPOCHS}ep_maxlen{MAX_LENGTH}_n{n_train}\n"
            f"best_optim_step: {best_step}\n"
            f"best_map3: {best[0]:.4f} [{best[1]:.4f},{best[2]:.4f}]\n"
            f"final_map3: {final[0]:.4f} [{final[1]:.4f},{final[2]:.4f}]\n"
            f"random_baseline: {random_baseline_map_at_k():.4f}\n"
            f"train_seconds: {train_seconds:.1f}\n"
            f"stopped_early_on_time_budget: {stopped_early}\n"
        )


def main() -> None:
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}, gpu count: {torch.cuda.device_count()}, random-guess loss={RANDOM_LOSS:.4f}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # `dtype=torch.float32` is LOAD-BEARING, not defensive. transformers 5.x
    # defaults from_pretrained to the CHECKPOINT's stored dtype, and both
    # deberta-v3-base and -large ship fp16 -- so the bare call returns fp16
    # PARAMETERS. That is not mixed precision (fp16 compute, fp32 master
    # weights); it is half-precision weights that AdamW updates in place, and it
    # silently cannot train: at lr=2e-5, an update is ~1.3 ULP for a weight near
    # 0.03 and BELOW HALF A ULP for any weight >= 0.1, so it rounds to nothing.
    # Measured on this recipe: max|delta| after one step was 1.072e-01 in fp16
    # (ULP-snapped garbage) versus a correct 2.001e-05 in fp32.
    # This is what pinned train_loss at ln(5)=1.6094 through four separate runs
    # while gradients, optimizer, labels, context and input format were all fine.
    model = AutoModelForMultipleChoice.from_pretrained(MODEL_NAME, dtype=torch.float32).to(device)
    bad = {p.dtype for p in model.parameters() if torch.finfo(p.dtype).bits < 32}
    if bad:
        raise RuntimeError(f"refusing to train {sorted(str(d) for d in bad)} parameters; see above")
    print(f"param dtype: {sorted(str(d) for d in {p.dtype for p in model.parameters()})}", flush=True)

    # Freeze embeddings + the first N encoder layers (cdeotte part 2's recipe).
    total_params = sum(p.numel() for p in model.parameters())
    if FREEZE_EMBEDDINGS:
        for p in model.deberta.embeddings.parameters():
            p.requires_grad = False
    layers = model.deberta.encoder.layer
    for layer in layers[:N_FROZEN_LAYERS]:
        for p in layer.parameters():
            p.requires_grad = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"froze embeddings={FREEZE_EMBEDDINGS} + first {N_FROZEN_LAYERS}/{len(layers)} layers: "
          f"{trainable/1e6:.1f}M of {total_params/1e6:.1f}M params trainable "
          f"({100*trainable/total_params:.1f}%)", flush=True)

    train_df = pd.read_parquet(find_data_file("train_pool_own_context_src2.parquet"))
    train_df["context"] = train_df["context"].str.slice(0, MAX_CONTEXT_CHARS)
    t1_df = pd.read_parquet(find_data_file("t1_dev_own_context_general_big.parquet"))
    t1_df["context"] = t1_df["context"].str.slice(0, MAX_CONTEXT_CHARS)
    t1_sel_df = t1_df.sample(n=min(EVAL_SUBSET_N, len(t1_df)), random_state=SEED).reset_index(drop=True)
    print(f"train: {len(train_df)} rows, T1 full: {len(t1_df)} rows, "
          f"T1 selection subset: {len(t1_sel_df)} rows")

    collator = DataCollatorForMultipleChoice(tokenizer)

    # Probe one real forward+backward at the chosen batch size and halve on OOM.
    # An unattended overnight run must not die on the first step.
    # Longest contexts first: padding is to the longest row in a batch, so peak
    # memory is set by the worst case, never by an arbitrary head() slice.
    worst_case_rows = train_df.assign(_L=train_df["context"].str.len()).sort_values(
        "_L", ascending=False).drop(columns="_L")
    global BATCH_SIZE
    while BATCH_SIZE > 1:
        try:
            probe = DataLoader(
                MultipleChoiceDataset(worst_case_rows.head(BATCH_SIZE), tokenizer,
                                      max_length=MAX_LENGTH, context_col="context"),
                batch_size=BATCH_SIZE, shuffle=False, collate_fn=collator)
            pb = {k: v.to(device) for k, v in next(iter(probe)).items()}
            with torch.autocast("cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
                probe_loss = model(**pb).loss
            probe_loss.backward()
            model.zero_grad(set_to_none=True)
            if device.type == "cuda":
                torch.cuda.empty_cache()
            print(f"batch probe OK at BATCH_SIZE={BATCH_SIZE}", flush=True)
            break
        except torch.OutOfMemoryError:
            model.zero_grad(set_to_none=True)
            if device.type == "cuda":
                torch.cuda.empty_cache()
            BATCH_SIZE //= 2
            GRAD_ACCUM_STEPS_NEW = 16 // BATCH_SIZE
            print(f"OOM at batch {BATCH_SIZE * 2} -> retrying batch {BATCH_SIZE} "
                  f"(accum {GRAD_ACCUM_STEPS_NEW}, effective batch 16)", flush=True)
            globals()["GRAD_ACCUM_STEPS"] = GRAD_ACCUM_STEPS_NEW

    train_loader = DataLoader(
        MultipleChoiceDataset(train_df, tokenizer, max_length=MAX_LENGTH, context_col="context"),
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collator,
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=LR, eps=1e-6, weight_decay=0.01
    )
    # PROPER mixed precision, which is the point of this whole run: fp32 MASTER
    # weights (so AdamW's ~lr-sized updates are representable -- see the dtype
    # note above) with fp16 COMPUTE (so activation memory and speed match the
    # old broken run, and batch 2 x 384 still fits a 16 GB T4). Loading fp32 and
    # training without autocast would roughly double activation memory and risk
    # OOM; loading fp16 makes the optimizer a no-op. Both halves are required.
    # autocast also keeps LayerNorm in fp32 automatically, which matters here:
    # deberta-v3's layer_norm_eps=1e-7 is a known fp16 NaN source on Turing.
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    num_optim_steps = (len(train_loader) // GRAD_ACCUM_STEPS) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.06 * num_optim_steps), num_training_steps=num_optim_steps
    )
    print(f"optim steps/epoch: {len(train_loader) // GRAD_ACCUM_STEPS}, total: {num_optim_steps}, lr={LR}")

    best = (-1.0, 0.0, 0.0)
    best_step = -1
    baseline = random_baseline_map_at_k()
    best_dir = OUT / "deberta-v3-large-src2-best"
    train_start = time.time()
    optim_step = 0
    stopped_early = False
    recent_losses: list[float] = []

    model.train()
    for epoch in range(EPOCHS):
        if stopped_early:
            break
        optimizer.zero_grad()
        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
                loss = model(**batch).loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at epoch {epoch + 1} step {step}")
            scaler.scale(loss / GRAD_ACCUM_STEPS).backward()
            recent_losses.append(loss.item())
            if (step + 1) % GRAD_ACCUM_STEPS == 0:
                # unscale before clipping, or the clip threshold is applied to
                # scaled gradients and effectively does nothing.
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)   # skips the step if grads are inf/nan
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                optim_step += 1
                if optim_step % EVAL_EVERY_STEPS == 0:
                    mean, lo, hi, _ = evaluate(model, tokenizer, t1_sel_df, collator, device)
                    loss_mean = sum(recent_losses) / len(recent_losses)
                    recent_losses = []
                    print(
                        f"ep{epoch + 1} step {optim_step}/{num_optim_steps} "
                        f"train_loss {loss_mean:.4f} (random {RANDOM_LOSS:.4f}, must FALL below it) "
                        f"T1sub MAP@3 {mean:.4f} [{lo:.4f},{hi:.4f}] base {baseline:.4f} "
                        f"[{time.time() - train_start:.0f}s]",
                        flush=True,
                    )
                    if mean > best[0]:
                        best, best_step = (mean, lo, hi), optim_step
                        best_dir.mkdir(parents=True, exist_ok=True)
                        model.save_pretrained(best_dir)
                        tokenizer.save_pretrained(best_dir)
                        write_summary(
                            OUT / "result_summary.txt", best, best, best_step,
                            len(train_df), time.time() - train_start, False,
                        )
                if time.time() - train_start > TIME_BUDGET_S:
                    print(f"TIME BUDGET reached at optim_step {optim_step} -- stopping gracefully", flush=True)
                    stopped_early = True
                    break

    train_seconds = time.time() - train_start

    # End-of-training weights on the FULL 1,500-row T1.
    final_mean, final_lo, final_hi, _ = evaluate(model, tokenizer, t1_df, collator, device)
    print(f"FINAL(end-of-training) full-T1 MAP@3: {final_mean:.4f} [{final_lo:.4f},{final_hi:.4f}]")

    # The best-by-subset checkpoint, re-scored on the FULL 1,500 rows -- the
    # headline number. Selection used the 500-row subset, so this full-set
    # rescore is what gets reported, never the subset figure.
    if best_dir.exists():
        del model
        torch.cuda.empty_cache()
        model = AutoModelForMultipleChoice.from_pretrained(best_dir).to(device)
        b_mean, b_lo, b_hi, _ = evaluate(model, tokenizer, t1_df, collator, device)
        print(f"BEST(step {best_step}) full-T1 MAP@3: {b_mean:.4f} [{b_lo:.4f},{b_hi:.4f}] base {baseline:.4f}")
        print(f"  (its 500-row selection score was {best[0]:.4f} [{best[1]:.4f},{best[2]:.4f}])")
        best = (b_mean, b_lo, b_hi)
    else:
        print("no best checkpoint was saved -- reporting end-of-training weights only")
        best = (final_mean, final_lo, final_hi)
    print(f"training took {train_seconds:.1f}s, stopped_early={stopped_early}")
    write_summary(
        OUT / "result_summary.txt", best, (final_mean, final_lo, final_hi),
        best_step, len(train_df), train_seconds, stopped_early,
    )


if __name__ == "__main__":
    main()
