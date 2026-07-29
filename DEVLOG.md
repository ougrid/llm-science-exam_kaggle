# Dev Log

A running, honest record of decisions, bugs, and reasoning as they actually
happened — the raw material for the interview walkthrough and for
`reports/limitations.md` and `reports/error_analysis.md` later. This is not
`experiments/log.csv` (that's the numeric run-by-run record); this is the
narrative of *why*.

Organized by the day numbering in `PLAN.md`'s 4-day core track. Entries are
chronological within each day.

---

## Day 0 — framing and setup

**Decision: this is a reproduction study, not a rank-chasing project.** The
competition closed in 2023; late submissions earn no medal. Framed instead
as measuring the retrieval-vs-scale question under hard offline constraints,
with retrieval measured independently of the reader. This is what makes the
official 200-row test set a feature (forcing honest statistics) rather than
an embarrassment — see `PLAN.md`'s validation design section.

**Decision: the 200 official rows are the held-out gold test set, never
trained on.** All training data comes from public synthetic sets. One
sentence to defend, and it's the single highest-leverage decision in the
project.

- Set up git identity, GitHub CLI, and a public repo
  (`github.com/ougrid/llm-science-exam_kaggle`) — public because the repo
  link is itself part of the interview pitch.
- Wrote `CLAUDE.md` encoding engineering conventions, borrowing genuinely
  transferable discipline (Conventional Commits, "numbers from measurement
  not intuition," "report honestly") from a prior DevOps take-home project,
  and writing this project's own non-negotiables (never train on gold,
  paired-bootstrap comparisons, capped gold-set evaluations) directly from
  `PLAN.md`'s validation design rather than forcing an analogy that didn't
  fit.

## Day 1 — foundations, the closed-book plateau, and a lot of debugging

### Environment and Kaggle platform facts

- `uv python install 3.11` + `uv venv` — sidesteps the broken system
  `python3-venv`/`ensurepip` on this box entirely.
- Installed `kaggle` CLI. Auth works via `~/.kaggle/access_token`, **not**
  the classic `username`+`key` `kaggle.json` — Kaggle's auth apparently
  moved to a bearer-token file read by a newer `kagglesdk` bundled with
  `kaggle==2.2.4`. Verified by actually calling the API rather than trusting
  either the old convention or a guess.
- **Verified the biggest open unknown: late submission to this closed
  competition still scores.** Pushed a dummy kernel emitting `"A B C"` for
  every row. Public 0.375156, Private 0.356882 — both close to the analytic
  random baseline of 0.3667 (small gap is real label-distribution
  imbalance across A-E, not a bug). This unblocks the entire project; without
  it there'd be no external signal at all.
- **Found and fixed a real bug in that first submission attempt**: the
  script assumed competition data mounts at `/kaggle/input/<slug>/`
  (the widely-documented 2023-era convention). It actually mounts at
  `/kaggle/input/competitions/<slug>/` — an extra path segment that isn't
  documented anywhere I found. Diagnosed by making the script self-report
  `os.listdir("/kaggle/input")` and glob for `test.csv` rather than guessing
  a second time.
- **Found: there is no CLI/SDK method to submit a kernel run to a
  competition.** Confirmed by reading `kagglesdk`'s actual
  `competition_api_service` source — no such call exists. The final "Submit
  to Competition" click is manual, in the browser, every time, for any code
  competition. Also hit a real UI bug along the way ("Could not find
  provided notebook 129057785") caused by trying to submit from a stale
  in-progress "Edit" draft instead of the completed kernel version — fixed
  by submitting the actual completed run instead.
- **Corrected a load-bearing wrong assumption: GPU quota is 6 hours/week
  (21600s), not the widely-cited 30.** Found by calling
  `kagglesdk`'s `get_accelerator_quota_statistics` directly rather than
  trusting community folklore that `PLAN.md` had (correctly) flagged as
  unverified. This is a 5x tighter budget and changes how the Kaggle
  `deberta-v3-large` run needs to be scoped (tight `--timeout`, log actual
  minutes used, check remaining quota before every push).

### Metrics and eval tiers

- Built `src/llmsci/metrics.py` first, before any model — `map_at_k`,
  `bootstrap_ci`, `paired_bootstrap`, `minimum_detectable_effect` — per
  `CLAUDE.md`'s own rule that every later accept/reject decision routes
  through this. 17 unit tests, including a check that a genuinely random
  submission's bootstrapped MAP@3 CI actually contains the analytic 0.3667.
- Built the three-tier eval split (`src/llmsci/data.py`,
  `scripts/build_eval_tiers.py`): T1 (1500-row synthetic dev), T2 (the
  official 200, read-only), T3 (1000-row OOD: ARC-Challenge + MMLU-STEM,
  normalized to a common schema with positional re-lettering since ARC's
  `answerKey` isn't always a letter).
- **Caught a leakage bug before it mattered**: the naive plan was to train
  the closed-book baseline on the full 6.5k synthetic pool, but 1500 of
  those exact rows had just been carved out into T1. Fixed by building an
  explicit `train_pool` (pool minus T1 minus gold near-duplicates) and
  training on that instead — otherwise T1 would have been contaminated as a
  selection set from day one.
- **Real finding, not a bug**: 16 rows in the public `radek1` 6.5k pool are
  near-duplicates of the official gold 200 (exact-Jaccard, threshold 0.8) —
  real contamination in a widely-used public training set, excluded before
  sampling.
- **Real finding, not a bug**: MMLU's own test set contains exact-duplicate
  question stems — a genuinely repeated item within `college_physics`, and
  a generic "Which statement is true?" stem shared verbatim across
  `elementary_mathematics` and `high_school_mathematics` (different options,
  identical stem text). Deduped by question text before sampling T3.

### The closed-book baseline: six attempts to get a trustworthy number

This is the most interview-relevant material from today — a real debugging
trail, not a single clean run. Every invalid run is logged in
`experiments/log.csv` with a `notes` field explaining exactly what was
wrong, rather than being deleted. **Nothing here was fixed by lowering
standards on what counts as a valid result — every fix was validated
against direct evidence before moving on.**

1. **Pure bf16 weights, AdamW directly on them, lr=1e-5.** Loss sat flat at
   ln(5)=1.6094 across two full epochs (1.6092 -> 1.6096, literally
   unmoved). Root cause: casting the whole model to bf16 and optimizing it
   directly makes AdamW's updates round to zero against bf16's ~3-digit
   mantissa at this learning rate. The MAP@3 that run produced (0.4241,
   CI excluding baseline) looked like a real result and would have been
   very easy to accept at face value — it wasn't; a model that isn't
   learning shouldn't beat baseline, and it turned out this was likely a
   T1-specific fluke on an essentially frozen head, not a real effect.
2. **fp32 weights + bf16 `autocast`.** Fixes the underflow, but loss went
   NaN by ~step 100. Confirmed every one of 202 checkpoint tensors was NaN.
   This is `PLAN.md`'s own flagged DeBERTa-v3 gotcha
   (`layer_norm_eps=1e-7` instability) — the plan anticipated it for fp16
   on a Kaggle T4, but it turns out bf16 `autocast` triggers the same
   failure locally.
3. **Plain fp32, no autocast at all**, to rule out precision as the cause
   entirely. Added a fail-fast NaN guard first (worth doing regardless —
   no reason to burn a full run before discovering corruption). Still went
   NaN, now within a *single* optimizer step. Isolated on a synthetic
   repeated batch, no scheduler, no shuffling: a perfectly ordinary
   pre-clip grad norm (3.69) still produced NaN by the second forward pass.
   This ruled out precision (bf16 vs fp32 made no difference) and pointed
   at something more fundamental.
4. **Root-caused it**: this is Adam's well-documented first-step
   instability — near-zero bias-corrected second moment on step 1 makes
   the effective update size behave unpredictably for small-gradient
   parameters, and DeBERTa-v3 is known to be sensitive to it. Fixed with
   `eps=1e-6` (Microsoft's own documented DeBERTa training recipe, vs
   PyTorch's default `1e-8`). Verified on the same isolated synthetic
   batch first — 20 clean steps, no NaN — before touching the real script.
5. **Stable, but still not learning.** With `eps=1e-6` and uniform
   `lr=1e-5`, no crash, but loss still didn't trend down over 2 epochs and
   T1's CI included the random baseline. Missed `PLAN.md`'s own
   hyperparameter table, which specifies *two* learning rates (encoder
   1e-5, head 1e-4) — the freshly-initialized classifier/pooler head needs
   a much bigger push than the pretrained body. Tried discriminative LR.
6. **Discriminative LR made it worse.** Direct weight-diff check confirmed
   parameters genuinely were updating (ruling out a "frozen model" theory
   that the suspiciously-identical epoch means, 1.6404 vs 1.6404, initially
   suggested) — but a no-warmup isolated replay on real data showed loss
   climbing monotonically at every head LR tried from 1e-5 to 5e-5,
   reaching 40+ at 5e-5. Genuine divergence, just dampened by warmup enough
   to not crash the full run outright.
7. **Settled on `PLAN.md`'s own cited community-converged recipe** instead
   of continuing to grid-search blind: uniform `lr=5e-6` (the low end of
   the 4e-6 to 8e-7 range `PLAN.md` cites from the 10th-place solution
   repo), `eps=1e-6`, gradient accumulation to reach their effective batch
   of 32, 3 epochs (bumped from 2 since the larger effective batch means
   fewer optimizer steps per epoch). **Result pending** — training in
   progress as of this entry; see `experiments/log.csv` for the outcome.

**The lesson worth stating out loud in the interview**: reduced precision
(bf16) was premature optimization on a 184M-parameter model with no real
memory pressure on 8GB VRAM — it cost two broken runs and zero benefit.
And a flat or NaN loss, or a suspiciously-clean number, is a prompt to
verify directly (weight diffs, isolated synthetic batches, checkpoint
inspection) rather than to accept or reject a result on the loss curve's
appearance alone.

---

<!-- Append new entries above this line as work continues. -->
