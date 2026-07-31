"""DAY 4: source-MATCHED training -- the actual diagnosis.

WHAT THE OVERNIGHT RUNS RULED OUT.
Two 5-hour runs (ougridd/night-large, ougridd/night-base) reached ~2,000
optimizer steps -- 3.3x the 600 steps of the previous attempt -- and moved
nothing: 0.3840 [0.3649,0.4036] and 0.3746 [0.3558,0.3937] against a 0.3667
baseline. So training VOLUME was not the gap, and the "starved optimization"
story was wrong.

WHAT IT ACTUALLY IS: a train/eval distribution mismatch I introduced myself.
Source distributions, measured:

    T1 dev (eval):     source 2 = 1,397/1,500 (93%),  source 1 = 103
    39,249-row pool:   source 2 =   4,586      (11.7%), source 3 = 14,824 (38%)

We trained on a pool that is 11.7% source-2 and evaluated on a set that is
93% source-2. cdeotte's `all_12_with_context2.csv` merges 12 generators whose
context and question styles differ, so this is a real covariate shift, not a
cosmetic one.

Worse, this project had ALREADY found and fixed this once: row 5 of
reports/ablation_table.md -- MAP@3 0.6086 [0.5880,0.6291], still the best
own-model result in the project -- was trained on a source-MATCHED slice of
exactly 4,586 rows, which is precisely source 2's count. When I scaled the
pool 4,978 -> 39,249 rows for "more data", I traded the distribution match for
volume and regressed the fix.

THIS RUN: the same source-matched 4,586 rows, but with OUR general-corpus BM25
context instead of cdeotte's (so train and eval retrieval match by
construction, per CLAUDE.md), frozen-layer recipe, and enough epochs that the
small pool is no longer the constraint -- 286 optimizer steps/epoch at
effective batch 16, so 12 epochs is ~3,400 steps.

Reference points to beat: our own best 0.3840 (does not clear baseline), and
row 5's source-matched 0.6086 on cdeotte's context. The public-reader anchor on
our context is 0.8600 on the clean gold 200, which is what says the retrieval
is fine and the reader path is where the loss is.
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
EPOCHS = 12  # 4,586 rows -> 286 optim steps/epoch; the pool is small, so epochs are cheap
EVAL_EVERY_STEPS = 100
LR = 2e-5  # cdeotte part 2's value, paired with the freezing below
SEED = 42
# Sized to FINISH, not to train as long as possible: a kernel still running at
# the deadline yields nothing, because `kaggle kernels output` only serves files
# from finished runs. 75 min of training + a final full eval lands well inside
# the window, and the graceful stop below always writes the best checkpoint.
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
    ds = MultipleChoiceDataset(df, tokenizer, max_length=MAX_LENGTH, context_col="context")
    loader = DataLoader(ds, batch_size=EVAL_BATCH_SIZE, shuffle=False, collate_fn=collator)
    all_logits = []
    with torch.no_grad():
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
            model(**pb).loss.backward()
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
            loss = model(**batch).loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at epoch {epoch + 1} step {step}")
            (loss / GRAD_ACCUM_STEPS).backward()
            recent_losses.append(loss.item())
            if (step + 1) % GRAD_ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                optim_step += 1
                if optim_step % EVAL_EVERY_STEPS == 0:
                    mean, lo, hi, _ = evaluate(model, tokenizer, t1_sel_df, collator, device)
                    loss_mean = sum(recent_losses) / len(recent_losses)
                    recent_losses = []
                    print(
                        f"ep{epoch + 1} step {optim_step}/{num_optim_steps} "
                        f"train_loss {loss_mean:.4f} (random {RANDOM_LOSS:.4f}) "
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
