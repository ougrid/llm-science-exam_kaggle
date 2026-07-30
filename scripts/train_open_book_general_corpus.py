"""Open-book DeBERTa-v3-base trained on the general-Wikipedia corpus's own
BM25 retrieval context, in place of the mbanaei STEM-only corpus (row 7 of
reports/ablation_table.md).

Motivated by scripts/compare_corpus_recall_paired.py's resolved finding:
the general corpus's answer-support recall@5 on the same T1 rows is +0.1393
[0.1167,0.1620] higher than mbanaei's, even sampled down to 1.6M chunks (vs
mbanaei's full 2.1M). This tests whether that recall gain translates to a
MAP@3 gain -- unlike the earlier phrase-match reranking experiment, where a
comparable recall@5 gain (+0.075) did NOT translate (see ablation_table.md's
row 8 note: "reranking measurably HURT the trained reader").

Reuses row 7's exact hyperparameters (batch=2, accum=16, maxlen=384,
lr=5e-6, eps=1e-6) -- a config already validated safe on this local 8GB
card, deliberately NOT the long-context run's batch=1/maxlen=768 config
that crashed with a transient CUDA error (see experiments/log.csv's
CRASHED-RUN-PARTIAL row).

Trained AND evaluated entirely locally, to avoid the Kaggle-vs-local
evaluation-environment discrepancy documented in DEVLOG.md and
ablation_table.md, and to keep this directly comparable to row 7's
locally-rescored number (0.4297 [0.4099,0.4496]).
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForMultipleChoice, AutoTokenizer, get_linear_schedule_with_warmup

from llmsci.experiment import git_sha, log_experiment
from llmsci.gpu_guard import cap_memory_fraction, probe_training_speed
from llmsci.metrics import average_precision_scores, bootstrap_ci, random_baseline_map_at_k
from llmsci.reader.mc import DataCollatorForMultipleChoice, MultipleChoiceDataset, logits_to_ranked_labels

DATA = Path("data")
BEST_CHECKPOINT_DIR = DATA / "checkpoints" / "deberta-v3-base-open-book-general-corpus-best"
MODEL_NAME = "microsoft/deberta-v3-base"
MAX_LENGTH = 384
MAX_CONTEXT_CHARS = 8_000
BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 16
EVAL_BATCH_SIZE = 8
EPOCHS = 3
EVAL_EVERY_STEPS = 15
LR = 5e-6
SEED = 42
MAX_MS_PER_STEP = 800  # measured ~310 ms/microbatch for batch=2,len=384; see scripts/train_open_book.py


def evaluate(model, tokenizer, df, collator, device):
    model.eval()
    ds = MultipleChoiceDataset(df, tokenizer, max_length=MAX_LENGTH, context_col="context")
    loader = DataLoader(ds, batch_size=EVAL_BATCH_SIZE, shuffle=False, collate_fn=collator)
    all_logits = []
    start = time.time()
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            all_logits.append(outputs.logits.float().cpu().numpy())
    seconds = time.time() - start
    logits = np.concatenate(all_logits, axis=0)
    y_pred = logits_to_ranked_labels(logits, k=3)
    scores = average_precision_scores(df["answer"].tolist(), y_pred, k=3)
    mean, lower, upper = bootstrap_ci(scores, n_resamples=10_000, seed=0)
    model.train()
    return mean, lower, upper, seconds


def main() -> None:
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    if device.type == "cuda":
        cap_memory_fraction(0.975)
        probe_training_speed(MODEL_NAME, BATCH_SIZE, 5, MAX_LENGTH, device, MAX_MS_PER_STEP)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForMultipleChoice.from_pretrained(MODEL_NAME).to(device)

    train_df = pd.read_parquet(DATA / "train_pool_own_context_general.parquet")
    train_df["context"] = train_df["context"].str.slice(0, MAX_CONTEXT_CHARS)
    t1_df = pd.read_parquet(DATA / "t1_dev_own_context_general.parquet")
    t1_df["context"] = t1_df["context"].str.slice(0, MAX_CONTEXT_CHARS)
    print(f"train: {len(train_df)} rows, T1+own-context (general corpus): {len(t1_df)} rows")

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
                    mean, lower, upper, _ = evaluate(model, tokenizer, t1_df, collator, device)
                    print(
                        f"epoch {epoch + 1} optim_step {optim_step} T1+own-ctx(general) MAP@3 {mean:.4f} "
                        f"[{lower:.4f},{upper:.4f}] (baseline {baseline:.4f})",
                        flush=True,
                    )
                    if mean > best_mean:
                        best_mean, best_lower, best_upper = mean, lower, upper
                        best_optim_step = optim_step
                        BEST_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
                        model.save_pretrained(BEST_CHECKPOINT_DIR)
                        tokenizer.save_pretrained(BEST_CHECKPOINT_DIR)
            if step % 500 == 0:
                print(f"epoch {epoch + 1} step {step}/{len(train_loader)} loss {loss.item():.4f}", flush=True)
        print(f"epoch {epoch + 1}/{EPOCHS} mean loss: {total_loss / len(train_loader):.4f}", flush=True)
    train_seconds = time.time() - train_start

    final_mean, final_lower, final_upper, eval_seconds = evaluate(model, tokenizer, t1_df, collator, device)
    print(f"FINAL T1+own-ctx(general) MAP@3: {final_mean:.4f} [{final_lower:.4f},{final_upper:.4f}]")
    print(f"BEST (optim_step {best_optim_step}) T1+own-ctx(general) MAP@3: {best_mean:.4f} "
          f"[{best_lower:.4f},{best_upper:.4f}] (baseline {baseline:.4f})")
    print(f"training took {train_seconds:.1f}s")

    log_experiment({
        "date": pd.Timestamp.now().isoformat(timespec="seconds"),
        "git_sha": git_sha(),
        "config": f"deberta-v3-base_open-book_GENERAL-CORPUS-OWN-RETRIEVAL_maxlen{MAX_LENGTH}_{EPOCHS}ep_lr{LR}_eps1e-6_n{len(train_df)}_LOCAL-ONLY",
        "tier": "T1+own-BM25-general-corpus-context",
        "n": len(t1_df),
        "map3_mean": round(best_mean, 4),
        "map3_ci_lower": round(best_lower, 4),
        "map3_ci_upper": round(best_upper, 4),
        "random_baseline": round(baseline, 4),
        "train_seconds": round(train_seconds, 1),
        "eval_seconds": round(eval_seconds, 1),
        "hypothesis": (
            "does the general-Wikipedia corpus's measured recall@5 gain over mbanaei's STEM-only "
            "corpus (+0.1393 [0.1167,0.1620], paired, see compare_corpus_recall_paired.py) translate "
            "into a MAP@3 gain for the trained reader, or does it fail to translate the way the "
            "phrase-match reranking recall gain did"
        ),
        "notes": (
            f"Same hyperparameters as row 7 of reports/ablation_table.md (batch=2, accum=16, "
            f"maxlen=384, lr=5e-6, eps=1e-6), same T1 rows, only the corpus/context differs "
            f"(general-Wikipedia 1.6M-chunk sample vs mbanaei's full 2.1M-chunk STEM-only corpus). "
            f"Trained and evaluated entirely locally for direct comparability with row 7's "
            f"locally-rescored 0.4297 [0.4099,0.4496] -- see this same comparison logged as a proper "
            f"paired bootstrap in a follow-up run against the row-7 checkpoint, not just these two "
            f"separate CIs."
        ),
    })


if __name__ == "__main__":
    main()
