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
   fewer optimizer steps per epoch). Result: no crash, loss trend only
   marginally downward (1.6169 -> 1.6157 -> 1.6132), T1 MAP@3 = 0.3888 with
   a 95% CI of [0.3698, 0.4076] — the lower bound clears the 0.3667 random
   baseline, but only by 0.0031. Too thin a margin to accept at face value
   given everything above.
8. **Added per-epoch T1 evaluation and re-ran at 8 epochs** specifically to
   settle whether run 7 was undertrained (and would keep improving) or
   capped (and would plateau or stay flat). Neither turned out to be quite
   right — **there is no learning signal at all**. Per-epoch loss means
   oscillate randomly around ln(5)=1.6094 with no trend across all 8
   epochs (1.6088 / 1.6140 / 1.6275 / 1.6172 / 1.6145 / 1.6115 / 1.6117 /
   1.6106), and per-epoch T1 MAP@3 bounces between 0.353 and 0.386,
   sometimes *below* the random baseline. Run 7's CI-barely-clears-baseline
   result was almost certainly a statistical fluke, not real signal — this
   longer, better-instrumented run makes that clear.

### Where the closed-book baseline stands: unresolved, not failed

This is a genuine open problem, not something to paper over. What's
**confirmed**, independent of the score itself:
- Training mechanics are sound: gradients demonstrably flow (direct
  weight-diff check on both the classifier head and an encoder layer, run
  4/5's diagnostic), no NaN across a full 8-epoch run, `eps=1e-6` and
  `lr=5e-6` are both independently validated as numerically stable in
  isolation.
- The absolute result (~0.38, statistically indistinguishable from the
  0.3667 random baseline) is real below `PLAN.md`'s own predicted ~0.58 for
  this exact model/config, by a wide enough margin that "just needs more
  epochs" has now been directly tested and ruled out.

**Three hypotheses, none tested yet — this is the actual next step, not a
concluded finding**:
1. The closed-book input format passes an *empty* first segment
   (`context=""`) into `tokenizer(first, second, ...)`, producing
   `[CLS][SEP]prompt+option[SEP]` — a degenerate structure a pretrained
   encoder that expects two populated segments may not handle well.
2. `truncation="only_first"` with an empty first segment may not actually
   be able to truncate anything (there's nothing in the first segment to
   cut), so unusually long `prompt+option` pairs could be silently passing
   through **longer** than `MAX_LENGTH=256` rather than being truncated —
   worth directly checking token length distributions against the cap.
3. 4982 training rows, each about a different, unrelated Wikipedia STEM
   topic, may simply be too little repeated signal for a 184M-parameter
   model to generalize from with *zero* context — unlike a task like
   sentiment classification where the same cues recur across examples, and
   more directly relevant to `PLAN.md`'s own core thesis: retrieval matters
   more than closed-book performance, so this closed-book number may
   matter less than it currently seems to.

**Decision: stopping here for tonight rather than pushing the
`deberta-v3-large` run to Kaggle.** The account has a scarce 6-hour/week
GPU quota (see Day-1 quota-correction entry above); pushing a large-model
training run against an undiagnosed problem in the smaller model risks
burning a meaningful fraction of that budget on the same issue. Every
invalid or inconclusive run (7 of them across today) is logged in
`experiments/log.csv` with a `notes` field, not deleted — the full
per-epoch trace for run 8 is in that file's notes column.

**Hypothesis 2 checked and ruled out.** The mechanism is real: an empty
first segment plus `truncation="only_first"` genuinely raises `Truncation
error: Sequence to truncate too short to respect the provided max_length`
when the second segment alone exceeds `max_length` — verified directly.
But it never fires on this data. Computed the actual token-length
distribution of all 24,910 real prompt+option choices in `train_pool.csv`
with `truncation=False`: median 33 tokens, max 145, **zero rows exceed
256**. These closed-book inputs are short — mostly just
`[CLS][SEP]question option[SEP]` — nowhere near the cap. Not the cause of
the no-learning result. (Side note, not a bug: median 33 vs
`MAX_LENGTH=256` means most of every batch is padding — worth remembering
for throughput later, irrelevant to correctness now.)

**Hypothesis 1 checked, and it directly led to the actual root cause.**
Compared the current empty-first-segment format against the standard
two-segment MC format (`prompt` as first, `option` alone as second) on a
600-row subset, 5 epochs, per-epoch eval. Both formats showed the *same*
pattern, ruling out the input format as the cause: T1 MAP@3 jumped to
**0.5556 / 0.5600** (format A / B) after just epoch 1 — far above anything
seen all night, close to `PLAN.md`'s expected ~0.58 — then **degraded back
toward baseline** over epochs 2-5 (down to 0.365-0.381).

That planted the real question: why did the full 8-epoch/4982-row run
never show this spike at all? Re-ran the full dataset with a short
2-epoch schedule and **mid-epoch** checkpoints (every 25% of an epoch,
not just at epoch boundaries) to get finer resolution:

```
epoch 0 (pre-train)  T1 MAP@3 0.3794
epoch 1 @ 25%         T1 MAP@3 0.5061   <- the peak
epoch 1 @ 50%         T1 MAP@3 0.3727   <- already collapsed
epoch 1 @ 75%         T1 MAP@3 0.3591
epoch 1 @ 100%        T1 MAP@3 0.3812
epoch 2 @ 25-100%     T1 MAP@3 0.376-0.382 (flat near baseline)
```

**Root cause, now clear**: the model genuinely learns a real, strong
signal very quickly (within the first ~38 optimizer steps, ~25% of a
single epoch) and then destabilizes and collapses back toward baseline
almost as fast — all *within* the granularity of a single epoch. Every
prior run tonight evaluated only at epoch boundaries, so every single one
of them measured the model *after* the collapse had already happened,
making it look like "no learning ever occurred" when in fact learning
happened and was then lost, invisibly, between checkpoints. This is not a
bug in the training code — it's a real optimization-dynamics finding
(the model overshoots past a good optimum and doesn't recover at this
LR/schedule), and the fix is standard: evaluate frequently and keep the
best checkpoint by validation score, rather than only the final one. This
is different from the best-of-N-configs cherry-picking `PLAN.md` warns
against — it's early-stopping / best-checkpoint selection within a single
run's own trajectory, exactly what `Trainer(load_best_model_at_end=True)`
does by default.

**Next**: rewrite `train_closed_book.py` to evaluate every N optimizer
steps (not just per-epoch) and track+save the best checkpoint by T1
MAP@3, and report *that* number — with its CI — as the actual closed-book
baseline, rather than the collapsed end-of-training state this whole
debugging trail was chasing all night.

**The lesson worth stating out loud in the interview**: reduced precision
(bf16) was premature optimization on a 184M-parameter model with no real
memory pressure on 8GB VRAM — it cost two broken runs and zero benefit.
A flat or NaN loss, or a suspiciously-clean number, is a prompt to verify
directly (weight diffs, isolated synthetic batches, checkpoint inspection,
per-epoch tracking) rather than accept or reject a result on the loss
curve's appearance alone — and the honest ending to a debugging session is
sometimes "here are three ruled-in hypotheses and zero ruled-out ones,"
not a clean resolution manufactured to have a tidy story before stopping.

### Closed-book baseline: the fix works, but it uncovers a lexical shortcut

Reran `train_closed_book.py` with the best-checkpoint fix (eval every 10
optimizer steps, save whenever T1 MAP@3 improves). Result, on the full
4982-row `train_pool`, 3 epochs:

- **Best checkpoint**: optim_step 30 of 465 (still inside the 46-step
  warmup) — T1 MAP@3 **0.5641 [0.5439, 0.5840]**.
- **Final (end-of-training) checkpoint**: T1 MAP@3 0.3830 [0.3640, 0.4021]
  — collapsed back to baseline, exactly like every prior run.

This confirms the root cause found earlier tonight: a real, sharp spike
almost immediately after training starts, followed by collapse to a noise
floor for the remaining ~94% of training. Per-epoch mean loss stays
essentially flat at ln(5)=1.609 throughout (1.6169 → 1.6151 → 1.6108), so
the collapse is invisible in the loss curve — only per-step MAP@3
tracking exposes it.

Before accepting 0.5641 as "the" Day-1 closed-book number, two concerns
needed checking:

1. **Selection bias.** The best checkpoint was chosen by evaluating T1
   ~46 times and taking the max — the same "best-of-N" optimism
   `PLAN.md` warns about, just within one run's trajectory instead of
   across configs. Worth flagging, though the observed gap over baseline
   (+0.197) is far larger than that bias alone could produce.
2. **Whether the spike is a shortcut, not knowledge.** The spike lands
   *during warmup*, before the model has seen much data, and `train_pool`
   / T1 are the same synthetic generation process — ripe conditions for a
   shared surface artifact rather than real learning. `PLAN.md` already
   names this exact risk as the Day-1 "options-only bias probe," not yet
   run tonight.

Ran the cheapest possible version of that probe: **no model, no
training** — just rank the five options by raw character length, longest
first, and score that against the true labels.

```
T1          MAP@3 0.4780 [0.4577, 0.4989]   (random baseline 0.3667)
train_pool  MAP@3 0.4783 [0.4673, 0.4897]
mean option length: correct 77.8-79.2 chars, incorrect 71.7-72.9 chars
answer-letter distribution: ~19-21% each (uniform -- not a positional bias)
```

**A zero-parameter length heuristic beats the fine-tuned model's
converged end-state (0.3830).** This is the sharper way to say it: three
epochs of fine-tuning don't just fail to learn — they actively destroy a
signal a dumb heuristic gets for free, without replacing it with
anything. The GPT-3.5-generated correct answers in this pool are
systematically longer and more hedged than the distractors it generates
alongside them, and that is enough on its own to clear the random
baseline by 30%.

**Reframing, honestly:** the transient 0.5641 spike very plausibly
reflects the model rapidly latching onto length-correlated token-count
features in the first few dozen steps, before optimizer dynamics move it
away from that shortcut and into a region that has learned nothing to
replace it with. This doesn't mean 0.5641 is invalid as a
checkpoint-selection result — it means it cannot be reported as a clean
"closed-book science knowledge" number without this caveat attached. Both
findings are logged in `experiments/log.csv` (rows `..._BEST-CKPT` and
`diagnostic_only_longest_option_heuristic_no_training`), with the second
row's notes explicitly cross-referencing the first.

**Why this is a good finding, not a bad one, for the interview
narrative**: quantifying a benchmark's own shortcut, and showing that
your model partially/possibly rides that shortcut rather than transcends
it, is exactly the kind of senior-level scrutiny `PLAN.md`'s validation
section was designed to produce. The honest headline for Day 1 is not
"closed-book DeBERTa reaches 0.56" — it's "closed-book DeBERTa's best
checkpoint reaches 0.56, but at least 0.48 of the gap over random is
explained by a length artifact in the benchmark's own generation process,
and the model's converged state doesn't even clear that artifact." Open
decision, not yet made: whether to build a length-debiased eval variant
to isolate any non-length-driven signal, or treat this as sufficient
Day-1 characterization and move to Day 2 retrieval, where real passage
context should swamp the artifact either way.

---

<!-- Append new entries above this line as work continues. -->
