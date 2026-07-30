"""SCORE PUSH (leg A, fast): deberta-v3-base, properly-trained.

WHY THIS RUN EXISTS -- the diagnosis that reframed the whole project, found
on the last day by scripts/diagnose_overfit_sanity.py:

Every training run in this project held its loss at ~1.61 = ln(5), the loss of
uniform predictions over 5 options, and every run's MAP@3 hovered near the
0.3667 random baseline with an apparent "peak at optim_step 15 then collapse".
That was never a real learning dynamic. A 64-row overfit test drove per-batch
loss to 0.0018, proving the training loop, labels, and truncation are all
correct -- the reader was simply DRASTICALLY UNDERTRAINED:
  - lr=5e-6 with 10% warmup over 465 optimizer steps puts the effective LR at
    step 15 (the reported "best checkpoint") at 1.63e-6, i.e. the model had
    barely moved off its random initialization. The randomly-initialized
    AutoModelForMultipleChoice head (classifier.weight, pooler.dense.*, per
    every run's own LOAD REPORT) cannot train at that rate.
  - 465 optimizer steps on 4,978 rows vs the public 0.82-0.86 solutions'
    thousands of steps at 2-4x the LR on ~60k rows: a 10-20x gap in total
    learning, sufficient to explain 0.43 vs 0.86 with no appeal to model size
    or corpus quality.

FIXES APPLIED HERE (all three at once, since each is independently supported
and there is no time left to ablate them one at a time -- stated plainly
rather than presented as a controlled experiment):
  1. lr 5e-6 -> 2e-5, the standard range for MC fine-tuning of this model.
  2. Training pool 4,978 -> 39,249 rows (the T1-excluded, null-dropped
     cdeotte 12-source pool, re-retrieved through our own general-corpus BM25
     index so train/test retrieval still match per CLAUDE.md).
  3. batch 2 -> 4 (accum 16 -> 8, same effective batch 32): a free ~2x
     wall-clock win on T4, and larger real batches also reduce the gradient
     noise that made high LR unstable at batch=2 in the diagnostic.
  => 2,452 optimizer steps at 4x the peak LR: ~20x the total learning of any
     run attempted today.

Retrieval context comes from the general-Wikipedia corpus, whose
answer-support recall@5 measured +0.1393 [0.1167,0.1620] over the STEM-only
mbanaei corpus on the same T1 rows (paired bootstrap, resolved -- see
scripts/compare_corpus_recall_paired.py).

TIME_BUDGET_S stops training gracefully and still writes results/checkpoint,
so a run cut short by the session limit is never a total loss.
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

MODEL_NAME = "microsoft/deberta-v3-base"
MAX_LENGTH = 384
MAX_CONTEXT_CHARS = 8_000
BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 8  # effective batch 32
EVAL_BATCH_SIZE = 8
EPOCHS = 2
EVAL_EVERY_STEPS = 50
LR = 2e-5
SEED = 42
TIME_BUDGET_S = 5.0 * 3600
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
            f"config: {MODEL_NAME}_GENERAL-CORPUS-BIGPOOL_lr{LR}_bs{BATCH_SIZE}x{GRAD_ACCUM_STEPS}_"
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
    model = AutoModelForMultipleChoice.from_pretrained(MODEL_NAME).to(device)

    train_df = pd.read_parquet(find_data_file("train_pool_own_context_general_big.parquet"))
    train_df["context"] = train_df["context"].str.slice(0, MAX_CONTEXT_CHARS)
    t1_df = pd.read_parquet(find_data_file("t1_dev_own_context_general_big.parquet"))
    t1_df["context"] = t1_df["context"].str.slice(0, MAX_CONTEXT_CHARS)
    print(f"train: {len(train_df)} rows, T1: {len(t1_df)} rows")

    collator = DataCollatorForMultipleChoice(tokenizer)
    train_loader = DataLoader(
        MultipleChoiceDataset(train_df, tokenizer, max_length=MAX_LENGTH, context_col="context"),
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collator,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, eps=1e-6)
    num_optim_steps = (len(train_loader) // GRAD_ACCUM_STEPS) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.06 * num_optim_steps), num_training_steps=num_optim_steps
    )
    print(f"optim steps/epoch: {len(train_loader) // GRAD_ACCUM_STEPS}, total: {num_optim_steps}, lr={LR}")

    best = (-1.0, 0.0, 0.0)
    best_step = -1
    baseline = random_baseline_map_at_k()
    best_dir = OUT / "deberta-v3-base-score-push-best"
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
                    mean, lo, hi, _ = evaluate(model, tokenizer, t1_df, collator, device)
                    loss_mean = sum(recent_losses) / len(recent_losses)
                    recent_losses = []
                    print(
                        f"ep{epoch + 1} step {optim_step}/{num_optim_steps} "
                        f"train_loss {loss_mean:.4f} (random {RANDOM_LOSS:.4f}) "
                        f"T1 MAP@3 {mean:.4f} [{lo:.4f},{hi:.4f}] base {baseline:.4f} "
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
    final_mean, final_lo, final_hi, _ = evaluate(model, tokenizer, t1_df, collator, device)
    print(f"FINAL T1 MAP@3: {final_mean:.4f} [{final_lo:.4f},{final_hi:.4f}]")
    print(f"BEST (step {best_step}) T1 MAP@3: {best[0]:.4f} [{best[1]:.4f},{best[2]:.4f}] base {baseline:.4f}")
    print(f"training took {train_seconds:.1f}s, stopped_early={stopped_early}")
    write_summary(
        OUT / "result_summary.txt", best, (final_mean, final_lo, final_hi),
        best_step, len(train_df), train_seconds, stopped_early,
    )


if __name__ == "__main__":
    main()
