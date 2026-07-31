"""Day-2 open-book training on Kaggle (2xT4), using our own single-retriever
context data (data/train_pool_own_context.parquet + t1_dev_own_context.parquet,
built locally by scripts/build_own_retrieval_context.py from a BM25 index over
a 20k-article corpus slice -- same retriever for train and eval, avoiding the
train/eval context-source mismatch found in cdeotte's merged 12-source data).

Self-contained (no src/llmsci import) since this kernel isn't attaching the
repo as a dataset yet -- the reader/metrics logic below is a direct copy of
src/llmsci/reader/mc.py and src/llmsci/metrics.py's relevant functions.
Plain fp32, no autocast: DeBERTa-v3's layer_norm_eps=1e-7 is a known fp16 NaN
source on T4, and this run isn't supervised interactively, so it isn't worth
the risk for a modest speed gain.
"""

from __future__ import annotations

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


def find_data_file(filename: str) -> Path:
    """Locate a dataset file under /kaggle/input regardless of nesting.

    Learned on Day 1: Kaggle's mount path under a declared dataset isn't
    always a flat root (e.g. /kaggle/input/competitions/<slug>/ for
    competition data) -- glob for the file instead of assuming a fixed path.
    """
    matches = glob(f"/kaggle/input/**/{filename}", recursive=True)
    if not matches:
        print("DEBUG: /kaggle/input contents:")
        for p in glob("/kaggle/input/**/*", recursive=True):
            print(" ", p)
        raise FileNotFoundError(f"{filename} not found anywhere under /kaggle/input")
    return Path(matches[0])
OPTION_COLUMNS = ["A", "B", "C", "D", "E"]
MODEL_NAME = "microsoft/deberta-v3-base"
MAX_LENGTH = 384
MAX_CONTEXT_CHARS = 8_000
BATCH_SIZE = 2  # matches the local RTX 5050 config that's known to fit; T4
GRAD_ACCUM_STEPS = 16  # OOM'd at batch=8 in DeBERTa's disentangled attention
EVAL_BATCH_SIZE = 8  # despite having more usable VRAM -- not worth another debug cycle
EPOCHS = 3
EVAL_EVERY_STEPS = 15
LR = 5e-6
SEED = 42


# ---- metrics (copy of src/llmsci/metrics.py) ----


def average_precision_at_k(actual: str, predicted: list[str], k: int = 3) -> float:
    predicted = predicted[:k]
    for i, p in enumerate(predicted):
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
        idx = rng.integers(0, n, n)
        means[i] = scores[idx].mean()
    lo = (1 - ci) / 2
    hi = 1 - lo
    return float(scores.mean()), float(np.quantile(means, lo)), float(np.quantile(means, hi))


def random_baseline_map_at_k(num_options: int = 5, k: int = 3) -> float:
    return sum(1.0 / i for i in range(1, k + 1)) / num_options


# ---- reader (copy of src/llmsci/reader/mc.py) ----


def build_choice_texts(prompt: str, options: list[str], context: str = ""):
    first = [context] * len(options)
    second = [f"{prompt} {opt}" for opt in options]
    return first, second


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
        flattened = [{k: v[i] for k, v in feature.items()} for feature in features for i in range(num_choices)]
        batch = self.tokenizer.pad(flattened, padding=self.padding, return_tensors="pt")
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        if has_labels:
            batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch


def logits_to_ranked_labels(logits: np.ndarray, k: int = 3) -> list[list[str]]:
    order = np.argsort(-logits, axis=1)
    return [[OPTION_COLUMNS[i] for i in row[:k]] for row in order]


# ---- training ----


def evaluate(model, tokenizer, df, collator, device):
    model.eval()
    ds = MultipleChoiceDataset(df, tokenizer, max_length=MAX_LENGTH, context_col="context")
    loader = DataLoader(ds, batch_size=EVAL_BATCH_SIZE, shuffle=False, collate_fn=collator)
    all_logits = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            all_logits.append(outputs.logits.float().cpu().numpy())
    logits = np.concatenate(all_logits, axis=0)
    y_pred = logits_to_ranked_labels(logits, k=3)
    scores = average_precision_scores(df["answer"].tolist(), y_pred, k=3)
    mean, lower, upper = bootstrap_ci(scores, n_resamples=10_000, seed=0)
    model.train()
    return mean, lower, upper


def main() -> None:
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}, gpu count: {torch.cuda.device_count()}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForMultipleChoice.from_pretrained(MODEL_NAME, dtype=torch.float32).to(device)
    # transformers 5.x follows the checkpoint dtype and deberta-v3 ships fp16, so
    # a bare from_pretrained yields fp16 PARAMETERS, which AdamW cannot move: at
    # lr=2e-5 an update is ~1.3 ULP near a weight of 0.03 and sub-half-ULP for any
    # weight >= 0.1. That pinned train_loss at ln(5) across four runs.
    bad = {p.dtype for p in model.parameters() if torch.finfo(p.dtype).bits < 32}
    if bad:
        raise RuntimeError(f"refusing to train {sorted(str(d) for d in bad)} parameters")

    train_df = pd.read_parquet(find_data_file("train_pool_own_context.parquet"))
    train_df["context"] = train_df["context"].str.slice(0, MAX_CONTEXT_CHARS)
    t1_df = pd.read_parquet(find_data_file("t1_dev_own_context.parquet"))
    t1_df["context"] = t1_df["context"].str.slice(0, MAX_CONTEXT_CHARS)
    print(f"train: {len(train_df)} rows, T1+own-context: {len(t1_df)} rows")

    collator = DataCollatorForMultipleChoice(tokenizer)
    train_ds = MultipleChoiceDataset(train_df, tokenizer, max_length=MAX_LENGTH, context_col="context")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collator)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, eps=1e-6)
    num_optim_steps = (len(train_loader) // GRAD_ACCUM_STEPS) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * num_optim_steps), num_training_steps=num_optim_steps
    )
    print(f"optimizer steps/epoch: {len(train_loader) // GRAD_ACCUM_STEPS}, total: {num_optim_steps}")

    best_mean, best_lower, best_upper, best_optim_step = -1.0, None, None, -1
    baseline = random_baseline_map_at_k()
    best_dir = OUT / "deberta-v3-base-open-book-own-retrieval-best"

    model.train()
    train_start = time.time()
    optim_step = 0
    for epoch in range(EPOCHS):
        total_loss = 0.0
        optimizer.zero_grad()
        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at epoch {epoch + 1} step {step}")
            (loss / GRAD_ACCUM_STEPS).backward()
            total_loss += loss.item()
            if (step + 1) % GRAD_ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                optim_step += 1
                if optim_step % EVAL_EVERY_STEPS == 0:
                    mean, lower, upper = evaluate(model, tokenizer, t1_df, collator, device)
                    print(
                        f"epoch {epoch + 1} optim_step {optim_step} T1+own-ctx MAP@3 {mean:.4f} "
                        f"[{lower:.4f},{upper:.4f}] (baseline {baseline:.4f})",
                        flush=True,
                    )
                    if mean > best_mean:
                        best_mean, best_lower, best_upper = mean, lower, upper
                        best_optim_step = optim_step
                        best_dir.mkdir(parents=True, exist_ok=True)
                        model.save_pretrained(best_dir)
                        tokenizer.save_pretrained(best_dir)
        print(f"epoch {epoch + 1}/{EPOCHS} mean loss: {total_loss / len(train_loader):.4f}", flush=True)
    train_seconds = time.time() - train_start

    final_mean, final_lower, final_upper = evaluate(model, tokenizer, t1_df, collator, device)
    print(f"FINAL T1+own-ctx MAP@3: {final_mean:.4f} [{final_lower:.4f},{final_upper:.4f}]")
    print(
        f"BEST (optim_step {best_optim_step}) T1+own-ctx MAP@3: {best_mean:.4f} "
        f"[{best_lower:.4f},{best_upper:.4f}] (baseline {baseline:.4f})"
    )
    print(f"training took {train_seconds:.1f}s")

    with open(OUT / "result_summary.txt", "w") as f:
        f.write(
            f"config: deberta-v3-base_open-book_OWN-RETRIEVAL_maxlen{MAX_LENGTH}_{EPOCHS}ep_lr{LR}_eps1e-6_n{len(train_df)}\n"
            f"best_optim_step: {best_optim_step}\n"
            f"best_map3: {best_mean:.4f} [{best_lower:.4f},{best_upper:.4f}]\n"
            f"final_map3: {final_mean:.4f} [{final_lower:.4f},{final_upper:.4f}]\n"
            f"random_baseline: {baseline:.4f}\n"
            f"train_seconds: {train_seconds:.1f}\n"
        )


if __name__ == "__main__":
    main()
