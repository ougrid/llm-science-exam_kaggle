"""SHORT LR confirmation on deberta-v3-LARGE before committing 3 h of quota.

Why this exists rather than jumping straight to a full run. A local 2x2 on
deberta-v3-BASE found the learning rate, not the data, to be the binding
constraint: same 1,024 source-matched rows, same seed, same 150 optimizer steps,
freeze 9/12, fp32 master weights + fp16 autocast --

    lr 2e-5 (the value we inherited):  1.6091 1.6077 1.6100 1.6080 1.6107 1.6093
    lr 1e-4:                           1.6102 1.6056 1.5755 1.5453 1.4777 1.4382

The first is noise within +-0.002 of ln(5)=1.6094. The second is monotone and
accelerating. lr=2e-5 came from cdeotte part 2 -- used there on ~60k rows with a
435M model fine-tuned differently -- and was never measured on OUR setup of 4,586
rows and 77.2M trainable parameters. Data scale differs 13x, trainable parameter
count 5.6x.

But base is NOT large, and that matters more than usual here:
deberta-v3-large is documented as unstable to fine-tune, with runs that sit at
chance depending on LR, warmup and seed. A 5x LR that helps a 12-layer base model
can plausibly destabilise a 24-layer large one. Confirming on the real model for
~45 min is strictly cheaper than discovering it 3 h into a full run -- which is
exactly the mistake that has already been made twice in this project.

DESIGN. Short arms, no checkpointing, one number each. Every arm is identical to
the real recipe (maxlen 384, batch 2 x accum 8, freeze 18/24, fp32 + autocast,
6% warmup) except the LR, and each gets STEPS_PER_ARM optimizer steps -- enough
that the base model's separation (visible by step 75) would show.

ATTEMPT 2. The first attempt ran four arms at 100 steps each and returned three
nulls (lr2e-5 1.6125, lr1e-4 1.6154, lr5e-5 1.6166) before OOMing on the fourth.
Both of those outcomes were my errors, not findings about large:

  * The OOM was a leak. `del model, opt` left `sched` and `scaler` holding
    references to the optimizer, so AdamW's exp_avg/exp_avg_sq were retained on
    every arm and accumulated to 14.55 of 14.56 GiB. Fixed below.
  * 100 steps was too few to conclude anything. Re-measured afterwards on the
    FULL 4,586-row pool with no row repetition, base's winning arm went:
    step 100 -> 1.6084 (flat), step 125 -> 1.5431, step 150 -> 1.4372. The first
    attempt stopped every arm at exactly the step where base was still flat.

So the pre-declared "no arm learns on large => switch the headline reader to base"
branch did fire on attempt 1, and invoking it would have been wrong: the branch
assumed base and large were compared in the same regime, and they were not (base
had run 2.34 epochs over a repeated 1,024-row sample, large 0.35 epochs over
4,586 distinct rows). Following a pre-declared rule whose premise has been
invalidated is not discipline. The rule gets re-run under a fair comparison.

READ IT LIKE THIS, declared before running:
  * An arm whose train_loss falls clearly below ln(5) and keeps falling wins ->
    launch the full run at that LR.
  * If 1e-4 diverges or NaNs while 5e-5 learns, that is the documented
    large-model instability and 5e-5 is the answer.
  * If NEITHER arm has moved by step 300 -- 2.4x the margin base needed on the
    same data -- then the finding is real and specific to the large checkpoint.
    At that point report it and make deberta-v3-base the headline reader rather
    than buying more quota. base at 1.4372 is a working reader; large at ln(5) is
    not, and a 184M-parameter reader that trains beats a 435M one that does not.

The MAP@3 at the end of each arm is a sanity read on a fixed 300-row T1 subset,
not a result: 300 rows gives a 95% CI half-width around +-0.055, so it cannot
resolve anything small. train_loss is the signal here.
"""

from __future__ import annotations

import gc
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




# --- confirmation-run knobs -------------------------------------------------
# Measured on the FULL 4,586-row pool (deberta-v3-base, no row repetition), the
# winning arm's trace was:
#     step 100 -> 1.6084   (still flat; indistinguishable from a null)
#     step 125 -> 1.5431   (breaks away)
#     step 150 -> 1.4372
# The first large attempt stopped every arm at exactly 100 steps, i.e. at or just
# before the point where base needed to keep going. Its three nulls are therefore
# consistent with "stopped too early" and cannot be read as "1e-4 fails on large".
# 300 steps gives 2.4x the margin base needed, which matters because large is 2x
# the depth and separation may come later still.
STEPS_PER_ARM = 300          # ~48 min/arm at the measured 9.5 s/step
EVAL_ROWS = 300              # sanity only; +-0.055 CI half-width, not a result
SECONDS_PER_STEP_EST = 9.5   # measured on the day4 rerun; used only for budgeting
EVAL_OVERHEAD_S = 240        # 300-row eval + model load
LOG_EVERY = 20
# (label, lr). Ordered cheapest-risk first so a NaN in a later arm still leaves
# earlier results on disk -- each arm writes its summary as soon as it finishes.
# (label, lr, n_frozen_layers). The local 2x2 on base showed BOTH knobs matter,
# with LR dominant -- the inherited recipe sat at the worst corner (lowest LR and
# most freezing). Measured on base at 150 steps, 1,024 rows, same seed:
#   lr2e-5 freeze9/12 -> 1.6093 (flat)      lr2e-5 freeze0/12 -> 1.5677 (slow)
#   lr1e-4 freeze9/12 -> 1.4382 (clear)
# So the arms below vary LR at the kernel's freezing depth, then test whether
# less freezing compounds with the higher LR. 18/24 on large is the same 75% as
# 9/12 on base; 12/24 halves it.
# Ordered MOST IMPORTANT FIRST. Each arm writes result_summary.txt the moment it
# finishes, so if the session is cut or a later arm OOMs, the arm that actually
# decides the next step is already on disk. That is why the first attempt still
# yielded three usable arms despite crashing.
# Dropped from the first attempt: lr2e-5 freeze18/24 (already measured null at 100
# steps -- re-running the control costs 48 min to re-learn something we know) and
# lr1e-4 freeze12/24 (base measured LESS freezing as strictly worse: 1.6129 vs
# 1.4372 at the same LR, so it is not a candidate and it was the OOM victim).
ARMS = [
    ("lr1e-4 freeze18/24 (base winner; THE question)", 1e-4, 18),
    ("lr5e-5 freeze18/24 (insurance if 1e-4 destabilises 24 layers)", 5e-5, 18),
]


def run_arm(label, lr, n_frozen, train_df, eval_df, tokenizer, collator, device):
    torch.manual_seed(SEED)
    model = AutoModelForMultipleChoice.from_pretrained(MODEL_NAME, dtype=torch.float32).to(device)
    bad = {p.dtype for p in model.parameters() if torch.finfo(p.dtype).bits < 32}
    if bad:
        raise RuntimeError(f"refusing to train {sorted(str(d) for d in bad)} parameters")
    for p in model.deberta.embeddings.parameters():
        p.requires_grad = False
    for layer in model.deberta.encoder.layer[:n_frozen]:
        for p in layer.parameters():
            p.requires_grad = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    loader = DataLoader(
        MultipleChoiceDataset(train_df, tokenizer, max_length=MAX_LENGTH, context_col="context"),
        batch_size=BATCH_SIZE, shuffle=True, collate_fn=collator,
        generator=torch.Generator().manual_seed(SEED),
    )
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr, eps=1e-6, weight_decay=0.01
    )
    sched = get_linear_schedule_with_warmup(
        opt, int(0.06 * STEPS_PER_ARM), STEPS_PER_ARM
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    print(f"\n=== {label}  lr={lr:.0e} freeze={n_frozen}/24  ({trainable/1e6:.1f}M trainable) ===", flush=True)
    model.train()
    trace, window, step, t0, diverged = [], [], 0, time.time(), False
    it = iter(loader)
    while step < STEPS_PER_ARM:
        opt.zero_grad(set_to_none=True)
        for _ in range(GRAD_ACCUM_STEPS):
            try:
                b = next(it)
            except StopIteration:
                it = iter(loader)
                b = next(it)
            b = {k: v.to(device) for k, v in b.items()}
            with torch.autocast("cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
                loss = model(**b).loss
            if not torch.isfinite(loss):
                # Do not abort the whole confirmation: record it and move on, so
                # the remaining arms still produce numbers.
                print(f"  NON-FINITE loss at step {step} -- arm diverged", flush=True)
                diverged = True
                break
            scaler.scale(loss / GRAD_ACCUM_STEPS).backward()
            window.append(loss.item())
        if diverged:
            break
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        step += 1
        if step % LOG_EVERY == 0:
            m = sum(window) / len(window)
            trace.append(m)
            window = []
            print(f"  step {step:3d}/{STEPS_PER_ARM}  train_loss {m:.4f} "
                  f"(ln5 {RANDOM_LOSS:.4f}, must FALL below)  [{time.time()-t0:.0f}s]", flush=True)

    if diverged or not trace:
        result = (label, lr, trace, None, "DIVERGED" if diverged else "NO DATA")
    else:
        mean, lo, hi, _ = evaluate(model, tokenizer, eval_df, collator, device)
        verdict = "LEARNS" if trace[-1] < RANDOM_LOSS - 0.05 else "null (at ln5)"
        print(f"  end: train_loss {trace[-1]:.4f}  T1({EVAL_ROWS}) MAP@3 "
              f"{mean:.4f} [{lo:.4f},{hi:.4f}] base {random_baseline_map_at_k():.4f}  -> {verdict}",
              flush=True)
        result = (label, lr, trace, (mean, lo, hi), verdict)
    if device.type == "cuda":
        print(f"  peak GPU: {torch.cuda.max_memory_allocated()/2**30:.2f} GiB", flush=True)
    # Free EVERYTHING that can reach the optimizer. `del model, opt` alone is not
    # enough and is what OOM'd arm 4 of the first attempt at 14.55/14.56 GiB:
    # `sched` holds a reference to `opt`, and `scaler` outlives it too, so AdamW's
    # exp_avg/exp_avg_sq tensors (2 floats per trainable parameter -- 617 MB at
    # 77.2M params, 1.2 GB at 12 unfrozen layers) were retained on every arm and
    # accumulated across the run.
    del loader, sched, scaler, opt, model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        print(f"  after cleanup: {torch.cuda.memory_allocated()/2**30:.2f} GiB still allocated",
              flush=True)
    return result


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}, ln(5)={RANDOM_LOSS:.4f}, baseline={random_baseline_map_at_k():.4f}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_df = pd.read_parquet(find_data_file("train_pool_own_context_src2.parquet"))
    train_df["context"] = train_df["context"].str.slice(0, MAX_CONTEXT_CHARS)
    t1 = pd.read_parquet(find_data_file("t1_dev_own_context_general_big.parquet"))
    t1["context"] = t1["context"].str.slice(0, MAX_CONTEXT_CHARS)
    eval_df = t1.sample(n=min(EVAL_ROWS, len(t1)), random_state=SEED).reset_index(drop=True)
    print(f"train {len(train_df)} rows, eval subset {len(eval_df)} rows, "
          f"{STEPS_PER_ARM} steps/arm, {len(ARMS)} arms", flush=True)
    collator = DataCollatorForMultipleChoice(tokenizer)

    results = []
    t_start = time.time()
    for label, lr, n_frozen in ARMS:
        # Never start an arm that cannot finish: a kernel killed mid-arm loses that
        # arm entirely, and `kaggle kernels output` only serves finished runs.
        est = STEPS_PER_ARM * SECONDS_PER_STEP_EST + EVAL_OVERHEAD_S
        if time.time() - t_start + est > TIME_BUDGET_S:
            print(f"\nSKIPPING '{label}': {est/60:.0f} min needed, "
                  f"{(TIME_BUDGET_S - (time.time() - t_start))/60:.0f} min left in budget",
                  flush=True)
            continue
        results.append(
            run_arm(label, lr, n_frozen, train_df, eval_df, tokenizer, collator, device)
        )
        # Rewrite after EVERY arm: a kernel killed by the session cap must still
        # leave harvestable results for the arms that did finish.
        with open(OUT / "result_summary.txt", "w") as f:
            f.write(f"LR CONFIRMATION on {MODEL_NAME}, {STEPS_PER_ARM} steps/arm, "
                    f"maxlen {MAX_LENGTH}, fp32+autocast, freezing varies by arm\n")
            f.write(f"random_loss_ln5: {RANDOM_LOSS:.4f}  baseline_map3: "
                    f"{random_baseline_map_at_k():.4f}\n\n")
            for lab, lr_, tr, ev, verd in results:
                f.write(f"{lab}\n  lr: {lr_:.0e}\n  loss_trace: "
                        f"{' '.join(f'{x:.4f}' for x in tr) if tr else '(none)'}\n")
                if ev:
                    f.write(f"  map3_{EVAL_ROWS}rows: {ev[0]:.4f} [{ev[1]:.4f},{ev[2]:.4f}]\n")
                f.write(f"  verdict: {verd}\n\n")

    print("\n" + "=" * 76)
    for lab, lr_, tr, ev, verd in results:
        m = f"{ev[0]:.4f}" if ev else "  --  "
        first = f"{tr[0]:.4f}" if tr else "  --  "
        last = f"{tr[-1]:.4f}" if tr else "  --  "
        print(f"{lab:<52} {first} -> {last}  MAP@3 {m}  {verd}")
    print("=" * 76)
    winners = [(lab, tr[-1]) for lab, _, tr, _, v in results if v == "LEARNS"]
    if winners:
        best = min(winners, key=lambda x: x[1])
        print(f"WINNER: {best[0]} (train_loss {best[1]:.4f}). Launch the full run at this LR.")
    else:
        print("NO ARM LEARNS ON LARGE while 1e-4 clearly learns on base. Do not buy more")
        print("quota for LR variants -- this is specific to the large checkpoint. Report it")
        print("and consider making deberta-v3-base the headline reader.")


if __name__ == "__main__":
    main()
