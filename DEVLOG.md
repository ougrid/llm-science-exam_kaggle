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

## Day 2 — retrieval

**Decision: use cdeotte's pre-retrieved context to get "the jump" number
before our own retriever exists**, per `PLAN.md`'s explicit Day-2 shortcut.
Downloaded `cdeotte/60k-data-with-context-v2` and found `all_12_with_context2.csv`
is 60,347 rows across 12 source datasets, each already carrying a retrieved
`context` column — separate from `train_with_context2.csv`, which turned out
to be the official gold 200 with context attached (verified: 200/200 prompts
match `data/train.csv` exactly) and is therefore off-limits for anything
beyond a capped, logged gold-set check later.

**Leakage check before touching this data**: 0 rows overlap the gold 200
(safe). But joining on `prompt` alone against T1 is NOT safe — 9 prompts in
the 60k file are shared by multiple rows with *different* options/answers
(duplicate question text from different source generations), which a
prompt-only join would match ambiguously. Joined on the full row (prompt +
all five options + answer) instead: exact 1-to-1,500 match, zero ambiguity.
Built `data/t1_dev_context.parquet` (T1's 1,500 rows with matching retrieved
context, for evaluating on the *same* retriever the open-book model trains
on) and `data/train_pool_context.parquet` (52,923 rows, T1-exact-matches and
full-row duplicates removed).

**Corpus**: downloaded `mbanaei/all-paraphs-parsed-expanded` — 2,101,279
paragraph rows across 276,559 STEM Wikipedia articles, already
paragraph-parsed (title, section, text). Simplified `PLAN.md`'s "3-sentence
window, stride 1" chunking to "keep each paragraph as one chunk unless it
exceeds 200 words, then split into non-overlapping 200-word windows" —
median paragraph is already ~130 tokens, so most paragraphs need no
splitting at all. Documented as a deliberate time-boxed simplification in
`src/llmsci/corpus.py`'s docstring, revisit only if Day-3 attribution shows
chunk-boundary loss. Built a 20k-article pilot slice (168,114 chunks) rather
than the full corpus, per `PLAN.md`'s "iterate fast first" guidance.

**Sparse retrieval**: flat (non-hierarchical) `bm25s` index, `method="lucene"`,
over the 20k-article slice — hierarchical article→chunk only becomes
necessary at the full ~2M-chunk scale. Query = prompt + all five options
(`PLAN.md`'s recommended variant — options are free query expansion). Smoke
test on a real T1 row: the target article ("Didymogenes", a niche algae
genus) isn't even in the 20k-article slice, so BM25 returned lexically
similar but topically wrong chunks (other genus/taxonomy articles). Expected
at 20k/276k ≈ 7% article coverage — a real recall measurement needs the full
corpus, which is Day-3 work; the slice is fine for the mismatch-row
diagnostic below, where retrieval *quality* isn't the point.

**Two runs launched in parallel**, both using the best-checkpoint-by-
validation-score pattern from Day 1 (not repeating that debugging lesson on
a second model):

1. `scripts/train_open_book.py` — pilot open-book training on a random
   15,000-row subset of `train_pool_context`, `deberta-v3-base`,
   `max_length=384` (up from 256; context needs room), evaluated against
   `t1_dev_context` (context from the same cdeotte pipeline). This is "the
   jump" row.
2. `scripts/run_mismatch_row.py` — the Day-1 closed-book **best** checkpoint,
   *not retrained*, fed our own BM25 top-5 context on T1, at the same
   `MAX_LENGTH=256` it trained with, to isolate the input-format shock from
   any context-length effect. Run on CPU deliberately, since the open-book
   training job was already using ~7.9/8.15 GB VRAM — no GPU contention.

Results pending — both still running as this entry is written.

### The WSL2 shared-memory trap: a 5-hour stall from silent VRAM overflow

Checked back on both jobs and found the open-book training had completed
only 200 of 936 optimizer steps in **5 hours 13 minutes** — on track for
~24 hours total, for a pilot run sized to take under 2. The CPU-only paired
mismatch job was similarly stalled: 2.5 hours without finishing even one of
two conditions.

Root-caused by profiling forward+backward+step in isolation with synthetic
data: `batch=4, max_length=384` reserved **8.98 GB**, exceeding the RTX 5050
Laptop's actual 8.55 GB VRAM (confirmed via `torch.cuda.get_device_properties`).
WSL2's CUDA driver does not raise `OutOfMemoryError` when an allocation
exceeds physical VRAM — it silently backs the overflow with system RAM
("shared GPU memory," accessed over PCIe instead of the card's own GDDR),
which is 15-25x slower and produces no error, warning, or unusual log
output. Everything *looked* fine: `nvidia-smi` showed normal-looking
utilization, the process was clearly doing GPU work, it was just doing it
at a fraction of real speed. This is a nastier failure mode than an OOM
crash precisely because nothing signals it — the training script would have
"succeeded" eventually, just a day later, with no indication anything had
gone wrong.

For comparison: the Day-1 closed-book run (`batch=8, max_length=256`) peaked
at 7,902/8,151 MiB — 97% of true VRAM, uncomfortably close to the same
cliff, but on the right side of it, which is why it ran at normal speed.

**Fix, in two parts:**
1. Reduced the open-book config to `batch=2, max_length=384, grad_accum=16`
   (same effective batch of 32) — benchmarked at 5.17 GB peak reserved, well
   inside budget.
2. Added `torch.cuda.set_per_process_memory_fraction(0.975)` to both
   training scripts. This caps PyTorch's allocator below the true card size,
   so any future config that would have overflowed into shared memory now
   raises a normal `CUDA out of memory` error immediately instead of
   degrading silently for hours. Verified directly: the old `batch=4,
   max_length=384` config now fails fast with a real OOM under this cap,
   exactly as intended.

**The lesson worth stating in the interview**: on a memory-constrained local
GPU, especially under WSL2, a slow-but-not-crashing run is not evidence a
config fits — it can be evidence of the opposite. `nvidia-smi` utilization
alone doesn't distinguish "computing normally" from "computing normally but
against system RAM instead of VRAM." The fix isn't more patience, it's a
hard memory ceiling that turns silent degradation into a loud, immediate
failure — the same "fail fast, verify don't assume" instinct as the eps=1e-6
and best-checkpoint fixes from Day 1, applied one layer lower in the stack.

Both jobs were killed and relaunched with the fixed config; results follow
in the next entry once they complete.

### Mid-session interruption, and the vanishing option

Both jobs died silently partway through — not from a bug, but from the
underlying session process itself restarting (the harness's background-task
tracking pointed at a new session UUID afterward; everything on disk,
including the git history and both scripts, was untouched). Relaunched both
cleanly under the new session with the gpu_guard-equipped scripts, which
paid off immediately: the startup probe printed `235 ms/step (limit 800) --
OK` before committing to the real run, real confirmation instead of a guess.

The relaunched open-book training then ran a full epoch (468 optimizer
steps) with **zero learning signal**: mean loss 1.6155 (≈ ln(5) = 1.6094,
the same "stuck" number from Day 1's failed closed-book attempts), MAP@3
never once leaving the 0.365-0.386 noise band around the 0.3667 baseline.
Unlike Day 1's dynamic (a sharp early spike, then collapse), this was flat
from step 1 — a different signature, worth checking rather than assuming
it was the same root cause.

Decoded a few actual tokenized training examples by hand rather than
guessing, and one option came out as literally `nan[SEP]` — a real `NaN`
in the options column, stringified into the model's input as if it were a
legitimate answer choice. Checked prevalence directly: **13,674 of 52,923
rows (25.8%) in `all_12_with_context2.csv` have at least one null option**.
T1 itself was unaffected (T1's options are always non-null, so the
exact-row join used to build `t1_dev_context.parquet` can only match rows
in the source file that also happen to have non-null options there) — which
is exactly why this survived the earlier leakage/alignment checks: those
checks verified *correctness* of the join, not *quality* of the joined
data, and nobody had looked at the training dataframe's raw contents
directly until now.

Fixed in `scripts/build_context_train_pool.py`: drop any row with a null
option before building `train_pool_context.parquet` (39,249 clean rows,
down from 52,923). Added a permanent regression test
(`test_train_pool_context_has_no_null_options`) rather than trusting this
gets checked by hand next time. Killed the contaminated training run
(no value in letting it finish) and relaunched on the clean pool.

**The lesson**: two data-quality passes (leakage exact-match, row-alignment
verification) both passed cleanly on this dataset, and it was still 25.8%
garbage in a column neither check was designed to look at. "The join is
correct" and "the joined data is clean" are different claims — decoding
and reading a handful of *actual* model inputs by hand, not just checking
shapes and merge keys, is what caught this, and it's the same instinct as
Day 1's "verify, don't assume" lesson, aimed at the data this time instead
of the optimizer.

### The jump that didn't happen (yet) -- three diagnostics deep

The corrected, clean-data open-book run finished: a full 2 epochs (936
optimizer steps) with mean loss flat at ~1.613-1.615 (≈ ln(5) = 1.6094)
throughout both epochs -- no trend at all, closely matching the
contaminated run's loss (1.6155 dirty vs 1.6145 clean). **The null-option
fix was necessary but not sufficient**: removing 25.8% garbage rows barely
moved the headline number. Best checkpoint (optim_step 900 of 936 -- near
the *end* of training, nothing like Day 1's early spike): T1+ctx MAP@3
0.3833 [0.3642, 0.4023], CI includes the 0.3667 baseline. Not resolved.

Two more diagnostics before accepting that conclusion at face value:

1. **Source-distribution mismatch, confirmed real.** Checked which of
   `all_12_with_context2.csv`'s 12 source datasets T1's exact-row-matched
   context actually comes from: 93% source=2, 7% source=1. But the 15k-row
   training subset was 88% sources 3/6/4/9/8/7 -- source=2 is only ~11.7%
   of the training pool, and sources 5/10/11/12 don't survive the
   null-option filter at all. T1 is evaluating the model almost entirely on
   a context "flavor" the model barely trained on.
2. **But it doesn't fully explain the flatness.** Evaluated both
   checkpoints on 800 held-out rows matching the *training* source mix
   instead of T1's: best-ckpt 0.3942 [0.3688,0.4204] (CI barely includes
   baseline), final-ckpt 0.4071 [0.3817,0.4342] (CI barely excludes it --
   the weakest possible "resolved" result). Even in-distribution, there's
   at most a marginal lift, nowhere near PLAN.md's ~0.82-0.86 expectation.

So: a real, confirmed distributional confound, that turns out not to be
the (whole) story. Launched one more decisive test rather than settling for
an ambiguous conclusion: `scripts/train_open_book_source_matched.py` trains
*and* evaluates on source=2 only (4,586 rows, matching T1's own dominant
source exactly), removing the mismatch variable entirely, with eval every
15 optimizer steps (finer than the 15k pilot's 50) specifically to avoid
missing a Day-1-style narrow transient spike. If this also comes back flat,
the problem is upstream of source-mismatch entirely -- most likely cdeotte's
merged-12-source context data being lower-quality or differently-formatted
than assumed, or something in the reader/config itself that closed-book's
recipe doesn't fully transfer to longer, context-bearing inputs. Result
pending.

**Where this leaves the project, given the time budget**: three real,
well-documented findings so far today (null-option data bug, confirmed
source-distribution mismatch, marginal-at-best in-distribution signal) and
zero confirmation yet of PLAN.md's headline "jump." This is itself an
honest, defensible interim state for the interview narrative -- CLAUDE.md's
own rule is to mark unresolved results as unresolved rather than round them
up, and a rigorous elimination trail (data bug fixed, distribution
mismatch found and quantified, in-distribution signal measured and found
weak, one more targeted test in flight) demonstrates the actual skill being
assessed even before a final number exists.

### Resolution: source-matching was the missing piece

The source-matched run (train and eval both `source=2`, 4,586 rows, 3
epochs, eval every 15 optimizer steps) confirms it decisively:

```
optim_step 15   T1+ctx MAP@3 0.6086 [0.5880, 0.6291]   <- the peak
optim_step 30   T1+ctx MAP@3 0.3814 [0.3622, 0.4003]   <- already collapsed
optim_step 45-420  0.365-0.385, flat, several evals producing byte-identical
                    scores -- the model fully saturated into a static state
FINAL (end of training)  0.3721 [0.3534, 0.3909]
```

**0.6086 clears the 0.3667 baseline by a wide, resolved margin, and beats
closed-book's own best (0.5641)** -- the largest, cleanest signal found in
this project so far. The dynamic is identical in shape to every other run
this project has produced (closed-book's day-1 spike, this run's spike):
sharp early learning within the first ~15-30 optimizer steps, then rapid
collapse back to baseline for the remainder of training, every single time,
regardless of model input (closed-book, cdeotte-context mismatched-source,
now cdeotte-context matched-source). That consistency is itself informative
-- this looks like a real property of this exact
recipe (`lr=5e-6, eps=1e-6, this warmup schedule, this model`) hitting a
sharp early optimum it can't stay in, not a fluke specific to any one
dataset.

**What the last several hours actually established, in order:**
1. `all_12_with_context2.csv` has a real 25.8%-prevalence null-option bug
   (fixed, regression-tested) -- necessary, not sufficient.
2. T1's context is 93% source=2 while the training pool is dominated by
   other sources -- a real, quantified train/eval mismatch.
3. Even in-distribution (matching the *training* mix), the signal was only
   marginal -- ruling mismatch out as the *sole* explanation.
4. Matching train and eval to the *same* source (removing every other
   confound) produced the strongest result in the whole project.

Put together: source (2) was likely the more "learnable" one of cdeotte's
12 sources for this reader at this scale, and the 15k-row pilot's
88%-other-sources training mix diluted whatever source=2-specific signal
exists, while also handing T1 an evaluation context distribution the model
had barely trained on. Not fully decomposed -- it's not proven that
source=2 alone is special versus "any single consistent source beats a
100k-row 12-source blend at this data scale" -- but not needed to be
decomposed further today given the time budget; the actionable, defensible
finding is that within-source consistency mattered more than data volume
at this scale, exactly mirroring CLAUDE.md's non-negotiable rule about
matching train/test retrievers, just discovered the hard way instead of
assumed.

**Honest framing for the interview writeup**: this is not yet "the jump" to
0.82-0.86 PLAN.md describes from a properly-trained open-book pipeline --
it's a best-checkpoint-selected transient peak on a 4,586-row single-source
slice of someone else's pre-retrieved context, evaluated on a 1500-row dev
set. The right way to state it: *"With retrieved context and best-checkpoint
selection, a 184M-parameter reader reached 0.61 MAP@3 on dev, versus a 0.56
closed-book ceiling and a 0.37 random baseline -- but the gain was
transient (collapsing within 15 optimizer steps) and appeared only once
train/eval context sources were matched, which took three rounds of
diagnosis to find."* That sentence, with its caveats intact, is a stronger
signal of engineering judgment than a clean 0.85 with no debugging story
behind it.

**Next, given the time budget**: decide whether to (a) push forward using
our own BM25 retrieval pipeline (already built) to construct a fully
consistent, single-retriever train/eval split at scale, which would give a
clean, defensible open-book number without cdeotte's 12-source confound at
all, or (b) treat today's findings (data bug, distribution-mismatch
discovery, and the confirmed source-matched signal) as sufficient Day-2
material and move to consolidating/writing up given time constraints.

### Moving to Kaggle: our own retrieval, fully self-consistent

Rather than continuing to lean on cdeotte's confounded 12-source data, built
a fully self-consistent alternative: ran our own BM25 index (the 20k-article
slice) over `train_pool.csv` and `t1_dev.csv` with the identical retriever
and top-5 query construction (`scripts/build_own_retrieval_context.py`) --
by construction there is no train/eval source mismatch to find, because
there's only one source. (Caught the same null-option class of bug here too,
much rarer: 4/4982 `train_pool` rows, dropped with a guard clause -- worth
noting that even our own clean-built `train_pool.csv` wasn't immune to a
handful of null options, so this is now something to check by default on
any new options dataframe, not just cdeotte's.)

Pushed a training kernel to Kaggle (`ougridd/day-2-open-book-training-own-retrieval`,
2xT4, GPU+internet enabled since this is a training run producing a
checkpoint artifact, not the final offline submission notebook) using this
data -- deberta-v3-base, same hard-won recipe (lr=5e-6, eps=1e-6, eval every
15 optimizer steps, best-checkpoint selection from the start this time,
not bolted on after an all-nighter of debugging). Self-contained script
(reader/metrics code copied inline rather than attaching `src/llmsci` as a
dataset yet) to keep the packaging fast given the time window. Result
pending -- this consumes real GPU quota (unlike the Day-1 dummy-baseline
kernel, which ran CPU-only), worth tracking against the 6h/week budget.

### The Kaggle run: three infra failures, then a clean negative result

Took four kernel versions to get a clean run, each failure a genuine new
lesson rather than a repeat:
1. **v1**: `FileNotFoundError` on the dataset path, even though
   `dataset_sources` was correctly declared and the dataset was "ready".
   Fixed defensively with a glob-based file finder
   (`glob("/kaggle/input/**/<filename>")`) instead of trusting a fixed
   mount path -- the same lesson as Day 1's `test.csv` mount-path surprise,
   now generalized into a reusable pattern.
2. **v2** (after adding `"machine_shape": "NvidiaTeslaT4"`): got assigned a
   Tesla P100 anyway, and the installed PyTorch build has dropped sm_60
   support entirely (`Minimum ... capability (7.0) - (12.0)`) -- so even a
   correctly-pathed run would have failed immediately on that hardware.
   Same glob fix carried forward; accelerator came back as T4x2 next try
   without further changes (Kaggle-side scheduling variance, not something
   fixed on our end).
3. **v3**: real `CUDA OutOfMemoryError` inside DeBERTa's disentangled
   attention at `batch=8, max_length=384` -- surprising on a 14.56 GB T4
   when the same model fit at `batch=8, max_length=256` in 7.9 GB locally,
   but relative-attention memory doesn't scale as simply as raw token
   count suggests. Fixed by reusing the exact `batch=2, accum=16` config
   already validated safe on the local 8 GB card, rather than spending a
   fourth cycle tuning Kaggle-specific batch size.
4. **v4**: ran clean, 1h32m, `deberta-v3-base` on 4,978 rows retrieved by
   our own BM25 index (20k-article slice) over both train and T1 -- fully
   self-consistent by construction. Result: best checkpoint (optim_step 45)
   **0.3866 [0.3673, 0.4057]** -- the CI lower bound clears the 0.3667
   baseline by only 0.0006. Essentially unresolved, and notably *not* the
   sharp-spike-then-collapse shape every other run today produced -- best
   and final are nearly identical (0.3866 vs 0.3823), suggesting there
   wasn't much of a peak to catch in the first place.

**What this means, read against today's other result**: train/eval
retriever consistency was necessary for the cdeotte source-matched run to
show its 0.6086 peak, but it is evidently not *sufficient* on its own --
this run has perfect consistency (literally the same retriever for both
sides) and still landed at baseline. The most likely explanation is
retrieval quality: the 20k-article slice covers only ~7% of the ~276k-
article corpus, so for a large fraction of T1 questions the right source
article simply isn't in the index at all (the earlier mismatch-row work
already found this directly -- the "Didymogenes" example retrieved
completely unrelated taxonomy articles because the true source wasn't
present). Consistent-but-irrelevant context has little to teach a reader.
This actually strengthens the project's central thesis rather than
undermining it: retrieval quality, not just architecture or
train/eval hygiene, is the dominant lever -- exactly what `PLAN.md` set out
to demonstrate, now shown by a controlled negative result instead of only
a positive one.

**Natural next step**: scale our own retrieval from the 20k-article slice
to the full ~276k-article corpus (the chunker and BM25 index code already
support this -- `scripts/build_corpus_slice.py`'s `N_ARTICLES` is the only
knob). That would give a fully self-consistent, first-party pipeline with
real recall, independent of cdeotte's data and its source-mixing
confound entirely.

### Day 2 wrap-up (interview-ready summary)

Given the time budget, stopping here for Day 2 rather than chasing the
full-corpus retrieval extension tonight. The condensed version, for
demo prep (full detail in the entries above and `reports/ablation_table.md`):

**What we built.** A streaming corpus chunker (`src/llmsci/corpus.py`), a
flat BM25 index over a 20k-article slice, a reusable open-book training
script with best-checkpoint selection baked in from the start, and a
working Kaggle training-kernel pipeline (dataset packaging, GPU+internet
kernel, path/accelerator/OOM issues all found and fixed).

**What we found, as a chain of honest corrections rather than a single
clean result:**
1. Public "pre-retrieved context" data isn't automatically trustworthy —
   25.8% of `cdeotte/60k-data-with-context-v2` has a null option that
   silently becomes the literal string `"nan"`. Found by decoding actual
   tokenized model inputs by hand, not by trusting shape/merge checks.
2. That dataset also has a real train/eval confound: it merges 12 source
   pipelines, and our eval set happened to draw overwhelmingly from a
   source underrepresented in a naive training split.
3. Once matched, a real (if transient) learning signal appeared: **0.6086
   MAP@3**, beating the closed-book ceiling (0.5641) and both far above
   the 0.3667 random baseline.
4. But a fully self-consistent version built from our *own* retrieval
   (no possible source mismatch, by construction) landed at only 0.3866 —
   because that retriever's recall is capped by a small corpus slice.
   **Consistency alone doesn't buy the jump; retrieval quality does.**

**The one sentence for the interview**: *"I chased a promising shortcut
(pre-retrieved public context), found it had a real data bug and a real
distribution mismatch, fixed both, got a strong-but-transient signal once
I matched sources — and then proved with my own retrieval pipeline that
consistency alone wasn't the answer, recall was. That's the same
retrieval-beats-everything thesis this whole project is testing, arrived
at through debugging instead of assumed from the start."*

**Honest limitations carried into Day 3+**: neither headline open-book
number (0.61 or 0.39) is a stable, validated operating point — both are
single-run best-checkpoint selections, not confirmed across seeds or on
T3. The full ~276k-article corpus was never tried. Dense retrieval,
hybrid fusion, reranking, and the oracle-context ceiling are all
unstarted. These are exactly Day 3's planned work, not new scope.

## Day 3 — planning

**Decision (2026-07-30, 10:49 AM): commit, then full-corpus retrieval, then
Day 3 — but reordered, not run sequentially as three separate blocks.**
Asked directly whether "commit → full corpus → finish Day 3" fits in one
afternoon and did the arithmetic honestly rather than assuming the
per-day plan's ~7h estimates still hold after a Day 2 that ran long: full
corpus build + index + context retrieval realistically costs ~45–90
minutes at a scale never tried before (the 20k-article slice → full
~276k-article corpus is a ~14x jump, and today has hit a new snag at every
single phase so far), plus another ~90-minute retraining run matching both
of today's actual measured runtimes (3581s and 5384s). That's 3–4 hours
before Day 3's full 7-hour scope even starts — doesn't fit in an
afternoon.

**The fix**: run Day 3's retrieval eval harness (`retrieve/eval.py`) *while
the retraining job runs*, not after it. The eval harness needs the
retriever, the corpus, and T1/T3's known source labels -- it has no
dependency on a trained reader, so it was never actually on the same
critical path as retraining; sequencing them back-to-back was pure habit,
not a real dependency. Recognizing that got back most of an otherwise-idle
hour.

**Scope for today, in order**: commit → scale corpus/BM25 index to full
~276k articles → kick off retraining on full-corpus own-retrieval context
→ build the eval harness concurrently → interpret both together. Dense
retrieval, hybrid fusion, and reranking (Day 3 items 2-3) are explicitly
deferred to a later session, not cut — matches `PLAN.md`'s own
long-standing "never cut the eval harness" cut-order almost exactly, just
arrived at through today's actual pace instead of assumed in advance.
Full reasoning recorded in `PLAN.md`'s Day 3 section under "today's
compressed execution plan" so the two documents don't drift.

## Day 3 — execution (full autonomy granted)

User handed off full responsibility and stepped away, asking to be able to
check progress from a phone (GitHub for commits, Kaggle for the live
training run). Proceeded through the reordered plan:

**Full corpus scale-up was much faster than estimated.** Chunking all
2,101,279 paragraph rows (→ 2,345,229 chunks across ~276k articles) took
**49 seconds**, not the 45-90 minutes budgeted. Building the BM25 index
over all 2.34M chunks took another 143 seconds, with one real scare: RSS
peaked at ~9.8 GB during tokenization (15 GiB machine, ~3.9 GB "available"
at the low point) before settling back to 6.7 GB once indexing finished --
worth knowing for next time, but it didn't actually OOM.

**A genuine operational mistake, caught and fixed.** Attaching context to
train_pool/T1 via the new full-corpus index seemed to die silently (no
error, process gone) on the first attempt -- diagnosed as a likely OOM and
relaunched. It turned out the *first* attempt was still alive the whole
time (a `pgrep` check gave a false negative), so the "fix" actually
launched a second, fully redundant process alongside the first, and the
two together pushed memory to a real edge (876 MB free). Caught via `ps
aux` showing two matching PIDs, not by anything failing loudly. Killed the
redundant second process; the first had in fact already completed
successfully. Lesson: a silent, ambiguous failure (no error, no obvious
completion) deserves direct verification (`ps aux`, not just a `pgrep`
pattern that can transiently miss) before concluding it died and retrying
-- retrying blind on an ambiguous signal is exactly how you end up running
the same expensive job twice by accident.

**Retrieval eval harness (Day 3 item 1) results, run concurrently with
kicking off the full-corpus retraining on Kaggle** (full numbers and
caveats in `reports/retrieval_eval.md`): T1 recall@5 = 0.481, recall@100 =
0.711 (plateaus well short of 1.0 -- real retrieval misses plus proxy
limitations, both named rather than hidden). T3-OOD recall@100 = 0.588,
notably lower than T1's despite similar recall@1 -- a real, measured
distribution shift between GPT-3.5-generated and human-written questions,
exactly what T3 was built to catch. Oracle-context ceiling and the 2×2
failure decomposition still need the trained reader's predictions --
pending the Kaggle run.

Full-corpus retraining kicked off on Kaggle (same kernel, new dataset
version with the full-corpus context replacing the 20k-slice pilot data,
same filenames so `script.py` needed no changes). Result pending.

### The full-corpus result — and the finding that actually explains the gap

Full-corpus retraining finished (5801.5s, ~97 min): best checkpoint
(optim_step 15) **0.3869 [0.3681, 0.4058]** — essentially identical to the
20k-slice pilot's 0.3866. Scaling retrieval coverage 14x (20k → 276k
articles) moved the training result by 0.0003. Not the outcome expected
after the eval harness measured real recall (0.481 @5, 0.711 @100) on this
same index.

Rather than accept "recall isn't the answer" as the final word, spent five
minutes hand-comparing cdeotte's context against our own full-corpus
retrieval for the same T1 rows (merged on prompt) — and the answer was
immediate, not statistical. Row 0 (a question about the genus
*Didymogenes*): cdeotte's context opens with **"Didymogenes is a genus of
green algae in the class Trebouxiophyceae"** — nearly verbatim the correct
answer text. Our own retrieval, even over the full corpus, returned
*Ochromonas* — a different genus entirely (the same failure already
flagged back in the Day-2 mismatch-row work, now confirmed to persist at
14x the corpus coverage). Row 2 (Big Mama Thornton): cdeotte's context
names her explicitly; ours retrieved a Chinese internet-censorship
neologism and an unrelated athlete also surnamed Thornton — fooled by
matching "Big" + "mama" + "Thornton" as three independent tokens rather
than one named entity.

**This reframes the whole day's investigation.** It isn't that our
retriever finds fewer relevant passages in aggregate (recall@5=0.48 says
otherwise) — it's that BM25's bag-of-words scoring has no phrase- or
entity-level understanding, and gets outranked by lexically-similar but
topically-wrong chunks on exactly the multi-word named entities this
benchmark is full of. That failure mode doesn't improve with more corpus
coverage, because the problem is *ranking* what's already there, not
*whether* the right article exists in the index. cdeotte's context being
this precise is itself informative: it's either derived closer to the
true generating article than any retrieval this project has built, or
comes from a genuinely stronger retrieval pipeline — this sample doesn't
distinguish which, and that's an honest open question, not a gap papered
over.

**The actionable takeaway for a next session**: the next lever isn't more
corpus coverage, it's phrase-aware or entity-aware retrieval — requiring
multi-word phrase matches to score highly, or adding a reranker that can
tell "Big Mama Thornton" from three unrelated tokens. Logged as a proper
diagnostic row in `experiments/log.csv`, not folded silently into the
training result's notes.

All results from today are now logged: the corpus/index build, the
retrieval eval harness (both proxies... rather, both tiers of Proxy B),
the full-corpus retrain, and this diagnostic. `reports/ablation_table.md`
and `reports/retrieval_eval.md` still need a pass to fold this row in —
next up.

## Day 3 — the reranker fix, and a parallel reference reproduction

User returned, granted full continued autonomy, and asked two things
directly: whether the plan ever called for following a top Kaggle
solution's path, and whether the modest scores so far meant it was time to
pivot toward one. Answered honestly rather than either dismissing the
concern or abandoning the project's own thesis: `PLAN.md` did plan for
reproducing one public solution, but narrowly, as a calibration anchor,
with its own anti-patterns section calling blind-forking "the one thing
that sinks the project." Agreed on two parallel tracks: fix the actual
diagnosed problem (BM25's lack of phrase/entity awareness) in the main
pipeline, and spawn an isolated agent to faithfully reproduce a top
solution as an explicitly separate, clearly-labeled comparison track.

**The cheap fix worked.** Before committing to a cross-encoder reranker,
tested the cheapest possible version: reorder BM25's top-50 candidates by
counting exact multi-word phrase overlaps with the query
(`src/llmsci/retrieve/rerank.py`), no new model. On a 200-row T1 sample:
recall@5 improved **0.510 → 0.585**. Before trusting that, tested and
ruled out a competing hypothesis first — that the *Didymogenes*/Big Mama
Thornton failures found earlier meant a genuine corpus-domain gap (a
STEM-only corpus can't contain a blues singer). Checked at scale: 0/100
random T1 rows had their answer keywords entirely absent from the corpus.
Those two examples were rare edge cases, not representative — the ranking
diagnosis holds. Rebuilt the training/eval context with reranking applied
and kicked off a retrain on Kaggle; result pending.

**The parallel reproduction agent found something more valuable than a
score.** Working in an isolated git worktree (`isolation: "worktree"`,
kept fully segregated from `src/llmsci/`, `experiments/log.csv`, and this
file per its own instructions), it faithfully reproduced
`cdeotte/how-to-train-open-book-model-part-2` — read the actual pulled
notebook, wrote up its architecture in prose first, then reimplemented
from that prose, never copy-pasting. Three findings now integrated (full
detail in `reference_reproduction/RESULTS.md`, committed as its own
labeled comparison track, not blended into this pipeline's numbers):

1. **The published notebook's literal configuration doesn't clear
   baseline either**: 0.3770 [0.3577, 0.3960] on our T1, CI containing
   0.3667. Its `NUM_TRAIN_SAMPLES=1_024` is explicitly a demo subset
   (1.7% of available data), and the notebook's own markdown conditions
   the quoted 0.915 CV claim on "adjusting the parameters." A top
   solution's *literal, as-published* artifact is not the same thing as
   its *tuned, leaderboard-scoring* configuration — worth remembering
   before assuming any public score transfers to a faithful rerun.
2. **PLAN.md's calibration-anchor idea is unsound for this competition,
   and is now struck.** The shipped `mgoksu` checkpoint scores 0.9170 on
   T1 — but 100% of T1's 1,500 prompts are inside its own training data
   (it trains on the same public radek1 files T1 was built from). Any
   strong 2023 public checkpoint was trained on the union of the public
   synthetic pools; any synthetic dev set built from those pools is
   contaminated as an eval target for it, by construction. This is a
   real methodology correction to the plan, not a footnote.
3. **The 0.37-vs-0.61 gap this project has been chasing all day is
   checkpoint selection, not architecture.** The reproduction's
   final-checkpoint 0.3770 is statistically indistinguishable from this
   pipeline's own final-checkpoint result on identical cdeotte context
   (0.3721 [0.3534, 0.3913]) — different model size, learning rate, and
   context budget, same collapsed outcome. `deberta-v3-large` at 2e-5
   with `load_best_model_at_end=False` lands exactly where
   `deberta-v3-base` at 5e-6 does. This reframes 0.6086 honestly: it is a
   real number, reached through a legitimate and standard technique
   (best-checkpoint selection), but it is not evidence this pipeline's
   architecture or recipe is doing something the published solution
   isn't — both hit the same transient-spike-then-collapse wall.

The agent also independently reproduced this project's own null-option
finding (22.7% in cdeotte's data by its count, vs. this project's 25.8% —
close enough to be the same phenomenon, counted independently) and the
`AdamW eps=1e-8` NaN-on-sm_120 finding, adding `fused=True` as a second
working fix alongside this project's own `eps=1e-6`. It also ran the same
reranker idea on its own pipeline and found the same small-but-real
pattern (+0.0153 [+0.0007, +0.0293] in-window recall) — then explicitly
declined to spend two ~33-minute training runs measuring a reader-level
delta its own eval couldn't resolve. That's a good model of judgment to
carry forward: report the measurement that answers the question, and
don't spend compute on a number you already know will be noise.

Both `PLAN.md` and `reports/ablation_table.md` updated to reflect all of
this. `reference_reproduction/` is explicitly not part of this pipeline's
own headline numbers — cited here for the findings, not the score.

### The reranked retrain landed — and revealed a bigger problem than reranking

Kaggle's reranked-context run finished: best checkpoint (optim_step 105 of
~465) **0.3947 [0.3756, 0.4139]**, and — encouragingly at first glance —
the collapse was much gentler than every other run today (final 0.3858
stayed close to best, instead of crashing back to baseline). Compared
against the prior full-corpus run's self-reported 0.3869, this read as
confirmation that the phrase-match rerank fix worked.

**It didn't survive a proper check.** Before trusting two separately-eyeballed
Kaggle CIs (exactly the mistake CLAUDE.md's paired-bootstrap rule exists to
prevent), downloaded both best checkpoints and ran a real paired comparison
locally. The two numbers didn't just fail to match Kaggle's — they
contradicted them outright:

```
                    Kaggle self-reported      Local re-evaluation (same checkpoint, same data)
no-rerank (row 7)   0.3869 [0.3681,0.4058]    0.4297 [0.4099,0.4496]
reranked  (row 8)   0.3947 [0.3756,0.4139]    0.3807 [0.3618,0.3997]
```

Locally, the paired bootstrap (reranked − no-rerank) is **−0.0490
[−0.0750, −0.0229], resolved** — reranking *hurt*, the opposite of what
Kaggle's own numbers suggested. Ruled out a bug in the comparison script
first: re-ran each checkpoint in total isolation (separate processes, no
possibility of cross-contamination between loading two models
back-to-back) and got identical numbers to the paired run. This is a real
environment effect, not a script error.

**Most likely cause, not fully confirmed given the time budget**: Kaggle's
docker image almost certainly ships a different `transformers` version
than this project's local venv (`5.14.1`) — script.py never printed the
installed version, so this isn't nailed down precisely, only inferred from
elimination. `DebertaV2ForMultipleChoice`'s disentangled-attention
implementation is exactly the kind of code that can behave differently
across library versions without erroring.

**What this actually costs the project, stated plainly**: every number in
`experiments/log.csv` and `reports/ablation_table.md` that was *trained on
Kaggle and self-scored there* (both full-corpus own-retrieval runs) is now
suspect for comparison against anything trained and scored locally — which
is everything else in this project. Worse, each Kaggle run's own
best-checkpoint *selection* happened using Kaggle's environment during
training, so the checkpoint saved as "best" may not be the checkpoint that
scores best locally — and there is no way to recover that retroactively
without re-running training with local-environment scoring at every eval
step, which the time budget doesn't allow today.

**What still stands, and why**: the retrieval-level finding that motivated
building the reranker in the first place — recall@5 rising from 0.510 to
0.585 on a 200-row sample — was measured with no trained model involved at
all, purely a property of the retriever and the eval harness's own code,
run once, locally. It's untouched by any of this. What's now honestly
unresolved is the *next* link in the chain: whether better retrieval
ranking actually improves what the reader learns. Today's evidence says no,
but that evidence is itself compromised by the environment issue, so the
correct label is "not demonstrated," not "disproven."

`reports/ablation_table.md` rows 7-8 corrected with the local numbers,
struck-through Kaggle originals kept visible rather than edited away, and
a prominent methodological note added rather than a quiet fix. Logged as
its own diagnostic row in `experiments/log.csv`
(`CRITICAL_diagnostic_kaggle_vs_local_eval_environment_discrepancy`) given
how much of today's Kaggle-trained work it touches.

**The lesson, again, in a new place**: this is the same instinct as Day
1's eps/best-checkpoint debugging and today's earlier corpus-ceiling
correction — a result that "looks right" on first read (two CIs each
individually clearing baseline, in the expected direction) still needs the
actual paired, same-environment check before it gets to be a claim. Two
separate Kaggle runs agreeing with each other's *own* self-scoring is not
independent confirmation of anything if both share the same measurement
bias.

## Last day — reading the top-2 writeups, and a quota correction

User pushed on three things at once, all fair given this is the last day:
whether model capacity was the actual bottleneck, whether more capable
models were available and unused, and a GPU quota figure that didn't match
what they saw in the Kaggle UI (25h7m available of 30h, not the 6h/week
this project had "corrected" itself to on Day 1).

**Quota, corrected again.** The `get_accelerator_quota_statistics` API's
`totalTimeAllowed: 21600s` field is the **per-session** cap (6h), not the
weekly total — confirmed against the UI directly, which shows 30h/week.
Day 1's "correction" over-corrected: the original 30h/week figure was
right all along, the API field was just misread. `PLAN.md` fixed again.
Lesson under the lesson: a number coming from an API call is not
automatically more trustworthy than a widely-cited figure just because
it's programmatic — this field's own name (`totalTimeAllowed`) doesn't
disambiguate per-session from per-week, and verifying "does this match
what I can independently observe" (the UI) is still worth doing even for
API-sourced numbers.

**Read both top-2 writeups directly** (JS-rendered pages, not fetchable by
plain HTTP -- used the `agent-browser` skill to actually load and read
them) rather than continuing to reason from `PLAN.md`'s older secondhand
citations. Directly answers "is the model bad": no. 2nd place
(`lytic`/`ivan sorokin`)'s *primary* reader is the exact same
`DebertaV2ForMultipleChoice` this project uses, and their retrieval +
learned-reranker system alone reaches **0.916 Private** before any LLM is
added — their own ablation table shows the DeBERTa-only system does
almost all the work; a Mistral leg adds ~0.015 on top of an already-strong
base. Model capacity was never the bottleneck; retrieval infrastructure
is, exactly as this project's own findings today (recall vs. ranking vs.
corpus coverage) already pointed to independently.

**Concrete, actionable gaps identified against our own pipeline:**
1. **Corpus**: 2nd place indexed *general* Wikipedia (`graelo/wikipedia`,
   all topics), not a STEM-filtered set. This is almost certainly the
   real explanation for the Didymogenes/Big Mama Thornton failures found
   earlier — `mbanaei`'s corpus structurally cannot contain non-STEM
   entities, and no amount of BM25/reranking tuning fixes an article
   that was never indexed. `PLAN.md` already names a general-Wikipedia
   alternative (`jjinho/wikipedia-20230701`) that was never used until now.
2. **A trained reranker, not a heuristic.** 2nd place's reranker is a
   `DebertaV2ForMultipleChoice`-shaped model trained to predict *which
   retrieved chunk is best*, using pseudo-labels from a teacher model
   (their own early `deberta-v3-large` checkpoint). This project's
   phrase-match rerank was a cheap first test of the same idea (recall@5
   0.510→0.585) — the real version of it is training a small model to do
   this, which is very achievable with what's already built.
3. **Train on retrieved context, not the true source context** — 1st
   place explicitly measured this and found training on ground-truth
   context scored *worse* than training on retrieved context. Direct,
   independent validation of `CLAUDE.md`'s train/eval-retriever-consistency
   rule, arrived at by a top team for the same reason this project adopted
   it.
4. Additional public datasets used by 2nd place that this project hasn't
   touched: `openbookqa`, `ai2_arc`, `qasc`, `sciq`, `eduqg_llm_formatted`.

**Decision**: with quota corrected (real headroom: ~25h this week) and the
corpus gap being the most concretely evidenced fix (it explains a failure
already found by hand, not just theorized), swap to the general-Wikipedia
corpus next, in parallel with the still-running local context-length
sweep (CPU-bound download/chunk/index work doesn't compete with that job's
GPU use).

---

## Day 3, late — the reader was never trained: `ln(5)` and the collapse of a day's worth of "findings"

**This is the most important entry in this log, and it invalidates several
earlier ones.**

The user pushed to prioritize score, which forced a question I had not asked
directly all project: *why is the reader barely above baseline no matter what
we feed it?* I had been answering "weak model / weak corpus / distraction",
and had built an increasingly elaborate story around a
"learn fast, forget fast" training dynamic — a sharp early MAP@3 spike
followed by collapse, reproduced in nearly every run (see the Day-1 and
Day-2 entries above, and `reports/ablation_table.md` rows 3 and 5).

Then I looked at the loss values instead of the MAP@3 values:

```
epoch 1/3 mean loss: 1.6233
epoch 2/3 mean loss: 1.6223
epoch 3/3 mean loss: 1.6138
ln(5)              = 1.6094
```

`ln(5)` is exactly the cross-entropy of a uniform distribution over 5
options. **Every training run in this project sat at random-guess loss from
start to finish.** The model never learned anything. There was no dynamic to
explain.

**The decisive test** (`scripts/diagnose_overfit_sanity.py`): can the loop
overfit 64 rows? A correct training loop must be able to drive loss to ~0 on
a handful of examples. Swept lr ∈ {5e-6, 2e-5, 5e-5}, 60 steps each:

| lr | first-10-step mean | last-10-step mean | min per-batch |
|---|---|---|---|
| 5e-6 | 1.6266 | 1.6562 | 0.6753 |
| 2e-5 | 2.4061 | 2.2969 | **0.1010** |
| 5e-5 | 11.2580 | 8.7964 | **0.0018** |

Reaching **0.0018** proves the loop, the label alignment, and
`truncation="only_first"` are all correct — gradients flow and the
randomly-initialized MC head can learn. (The high *means* at high lr are
small-batch instability: this diagnostic used raw batch=2 with no gradient
accumulation, so effective batch was 2, not the production 32.)

So the cause was optimization scale alone. The arithmetic that should have
been checked on Day 1:

```
lr=5e-6, 10% warmup, 465 total optimizer steps
  -> effective LR at optim_step 15 = 1.63e-6
```

Optimizer step 15 is the step nearly every run in this project selected as
its "best checkpoint". At 1.63e-6 the model had barely moved off its random
initialization. Public 0.82–0.86 solutions used ~60k rows × 2 epochs at
1e-5–3e-5: **10–20× more total learning.**

**What this costs us, stated plainly rather than buried:**

1. **"Learn fast, forget fast" is not a finding.** It is noise around an
   untrained model. Every entry above describing it as a real dynamic
   (including the extended Day-1 investigation into whether the spike was a
   shortcut) was explaining an artifact. Those entries stay, with this
   correction pointing at them — deleting them would hide the actual
   mistake, which was reasoning elaborately about a number instead of
   sanity-checking that number's floor.
2. **Row 3's 0.5641 and row 5's 0.6086 are best-of-N selections over
   noise**, not learned performance. Row 1's options-only probe (0.4780) —
   a genuine zero-parameter lexical shortcut — is the more parsimonious
   explanation for anything above baseline.
3. **Every negative result about the reader is withdrawn, not refuted.**
   Phrase-match reranking "measurably HURT" (−0.0490), context length
   384→768 "measurably hurt" (−0.0503), the general-corpus swap "didn't
   translate" — all three were measured against a reader incapable of
   learning. They are now **NOT YET TESTED**. Three of the day's
   cleanest-looking resolved results evaporate at once.
4. **The retrieval-side results survive**, because they never depended on a
   trained reader: the corpus swap's +0.1393 [0.1167, 0.1620] recall@5 gain,
   the recall@k curves, and today's per-option-RRF result (−0.0780
   [−0.0960, −0.0607]: RRF *hurts* here, at 4× the query cost — PLAN.md
   called this its highest-expected-value trick, but also correctly
   predicted muted gains, because our pooled query already contains all five
   option texts, which is a free HyDE).

**The lesson worth carrying, and the reason this is the entry I would
actually tell in an interview:** I had a metric floor (0.3667) and a loss
floor (`ln(5)`) available from Day 1, and I monitored only the first. A model
pinned at its loss floor cannot have learned, no matter what the task metric
does — and task metrics on 1,500 rows are noisy enough to manufacture a
compelling narrative out of nothing. The cheapest guard against a whole day
of this is the overfit test, which takes about three minutes, and which I ran
for the first time on the last day.

**Action:** relaunched both legs with corrected optimization —
`ougridd/day3-score-push-base` (lr 2e-5, 39,249 rows, batch 4 × accum 8,
2 epochs ≈ 2,452 optimizer steps) and `ougridd/day3-score-push-large`
(`deberta-v3-large`, lr 1e-5, batch 2 × accum 16). Both now log train loss
alongside MAP@3 at every eval, specifically so the `ln(5)` floor is visible
in the output this time; both select checkpoints on a fixed 500-row T1 subset
and re-score the winner on the full 1,500; and both carry a `TIME_BUDGET_S`
graceful stop sized so the kernel actually *finishes* — Kaggle only serves
output files from finished runs, so a kernel still running at the deadline
would have yielded nothing at all. That last detail was itself a scheduling
bug I caught only by doing the eval-cost arithmetic (24 full-T1 evals on
`deberta-v3-large` ≈ 2 h of pure evaluation, which would have consumed the
entire remaining window).

---

## Day 3, later — correcting the correction: what the overfit test did and did not prove

Within the same hour of writing the `ln(5)` entry above, two things came back
that force amendments to it. Recording them here rather than editing that
entry silently, because the sequence of being wrong is the useful part.

**Amendment 1 — the overfit test does not prove the labels are meaningful.**
I wrote that reaching a per-batch loss of 0.0018 on 64 rows "proves the loop,
the label alignment, and `truncation='only_first'` are all correct". The loop
and the gradient path, yes. **Label alignment, no** — a model can memorize 64
rows perfectly even if their labels are randomly scrambled, because
memorization does not require the labels to mean anything. That inference was
simply invalid, and `ln(5)`-pinned loss is *exactly* what scrambled labels
would also produce, so the two hypotheses were still live and I had claimed
one was closed.

What actually validates the labels is a separate check: does the known
length shortcut show up in the training pool? Correct answers in this
GPT-3.5-generated benchmark run longer than distractors (row 1's options-only
probe measured 0.4780 vs a 0.3667 baseline). Measured on the pools directly:

| pool | longest option == answer | chance |
|---|---|---|
| `train_pool_own_context_general_big` (39,249 rows) | **32.7%** | 20% |
| `t1_dev_own_context_general_big` (1,500 rows) | **33.3%** | 20% |

Well above chance in both, so options and `answer` are correctly aligned and
the label hypothesis is genuinely closed now — by evidence that actually
bears on it. (Incidental finding from the same check, worth its own note:
**8.8% of training-pool rows contain duplicate option text**, which makes
those rows partly ill-posed. T1 has none. Not the main problem, but it is
free label noise in training.)

**Amendment 2 — "just raise the LR" is not sufficient, and I over-claimed it.**
`scripts/validate_lr_fix_local.py` ran the corrected recipe locally at
production-equivalent effective batch 32, lr=2e-5, 6,000 rows, 200 optimizer
steps. Loss went 1.618 → 1.826 (warmup spike) → drifting down → and only
crossed below `ln(5)` at step 150, at 1.6016, once the LR had decayed to
~5e-6. Two hundred steps of the corrected recipe produced almost no learning.

The honest reading is **not** that the LR diagnosis was wrong — the
1.63e-6-at-the-selected-step arithmetic still stands, and the effect is real.
It is that LR was necessary but nowhere near sufficient, and I presented a
partial cause as the whole cause. The probe is also not a faithful test of
the Kaggle runs: its LR schedule compresses into 200 steps whereas theirs
span 2,452 and 1,226, so its late-decay learning is a schedule artifact.

**What the probe did usefully surface: we never froze anything.**
`reference_reproduction/RESULTS.md` records that
`cdeotte/how-to-train-open-book-model-part-2` — the notebook behind the
published 0.823761 — froze **the embeddings and the first 18 of 24 layers**,
leaving 77.2M of 435.1M params trainable, at lr 2e-5 and effective batch 16.
Every run in this project trained all parameters. On a T4 at batch 2, a
full-unfreeze `deberta-v3-large` gets through only ~420 of its 1,226 planned
steps inside a 75-minute budget — the same starved regime the local probe sat
in. Freezing removes the backward pass and the AdamW state for three quarters
of the network, converting budget into real optimizer steps, and it is a
sample-efficiency win on a small pool. Swapped the full-unfreeze large leg for
`ougridd/day3-score-push-frozen` on that basis.

**Third finding, independent of training: we were only ever showing the reader
a quarter of its context.** Measured with the real tokenizer on 200 T1 rows:
retrieved 5-chunk context is a median **1,304 tokens**, prompt+longest-option
is a median 32, so at `max_length=384` the reader sees a median **26.8%** of
the retrieved context and the full context fits for **1.0%** of rows
(512 → 36.6%; 768 → 56.2%; 1024 → 75.7%). So the effective retrieval quality
the reader experiences is not recall@5 (0.6207) but something near recall@1
(0.4300) — answer-supporting chunks at ranks 2–5 are usually truncated away
before the model can read them. This makes the withdrawn
"context length 384→768 measurably hurt" result doubly suspect: it was
measured against an untrained reader *and* it is the change that should
mechanically have helped most.

**Net position going into the last two hours:** the reader's failure is
over-determined — starved optimization, no layer freezing, and a context
budget that discards three quarters of the retrieved evidence. Each is
independently supported and each is a live lever. What I no longer believe is
the tidy single-cause story I wrote an hour ago.

---

## Day 3, latest — I was wrong about `ln(5)`: loss at the uniform floor means UNCALIBRATED, not UNLEARNED

Third revision in one evening, and this one retracts the central claim of the
two entries above. Recording it in full because the error is instructive and
because leaving the wrong version standing would poison the artifact.

**The contradiction that exposed it.** The Day-1 closed-book run's per-epoch
mean loss was flat at `ln(5)` (1.6169 → 1.6151 → 1.6108) — which I had just
declared proof that "the reader never learned anything". But that same run
scored **MAP@3 0.5641 [0.5439, 0.5840]** at optimizer step 30. With n=1,500
and a per-row AP@3 SD of ~0.3, the standard error is ~0.008, so 0.5641 sits
roughly **25 standard errors above the 0.3667 baseline**. That is not noise,
and no amount of best-of-N selection over 1,500 rows manufactures a 25-SE
excursion.

**Why both facts are true at once.** MAP@3 depends only on the *rank order* of
the five logits; cross-entropy depends on their *magnitudes*. A model whose
logits are nearly equal — but consistently ordered so the correct option edges
ahead — has a loss arbitrarily close to `ln(5)` while ranking far better than
chance. `PLAN.md` states this exact property in its metric section
("monotone rescaling of a *single* model's scores is a no-op for MAP@3") and I
did not connect it to the loss curve I was interpreting.

So `loss ≈ ln(5)` means the reader is **badly calibrated / low-confidence**,
NOT that it failed to learn. My inference "loss at the floor ⟹ no learning"
was simply invalid.

**What this retracts (from my own entries earlier tonight):**

- **"The reader was never trained" — retracted.** It trains, weakly and
  without confidence. `ln(5)` was never evidence of the contrary.
- **"'Learn fast, forget fast' is not a finding" — retracted; the finding is
  restored.** A 25-SE peak at step 30 followed by a genuine fall to 0.3830 at
  end-of-training is a real trajectory, not selection noise. I over-corrected,
  and the struck-through text in `reports/limitations.md` and the banner in
  `reports/ablation_table.md` need reinstating with their original meaning.
- **My local probe was mis-designed.** `scripts/validate_lr_fix_local.py`
  deliberately measured only loss ("the loss floor is the thing being
  tested") and printed `VERDICT: STILL STUCK near ln(5)`. Given the above,
  that verdict is uninformative about learning: I needed MAP@3, the metric
  that is actually rank-based, and I explicitly chose not to compute it to
  save time. Its conclusion should be disregarded.

**What survives, and it is the most valuable measurement of the project.**
While the above was unravelling, the gold-200 reference eval finished:

> `mgoksu/llm-science-run-context-2`, a public 2023 checkpoint, fed **our own**
> general-corpus BM25 top-5 context, scores **MAP@3 0.8592 [0.8200, 0.8958]**
> on the clean gold 200 (n=200, baseline 0.3667).

This is a clean number (that checkpoint's training pools have zero prompt
overlap with the gold 200, asserted in `scripts/build_context_train_pool.py`),
and it is a *lower* bound, since the checkpoint reads context from a retriever
it was never trained against. It settles the attribution question this whole
project exists to answer:

**Our retrieval is not the bottleneck. Our reader is.** The same context that
yields 0.43 with our reader yields 0.8592 with a well-trained one. Corpus
scope, BM25 ranking, per-option RRF, and the 26.8% truncation budget are all
therefore second-order for us right now — a strong reader extracts 0.86 from
exactly the context we already produce. Every remaining point of score is in
reader training: layer freezing, LR schedule, training volume, and
calibration.

**The meta-lesson, which is the honest headline of this project.** In one
evening I asserted three incompatible root causes — undertrained reader,
over-determined failure, and now miscalibrated-but-learning — each with
confident supporting arithmetic. The failure mode was not lack of rigor in any
single step; it was reaching for a single-cause explanation and then
marshalling evidence for it, rather than looking for the measurement that
would discriminate between hypotheses. The measurement that finally did
(a known-good reader on our own context) was cheap, available all along, and
is exactly the "calibration anchor" `PLAN.md` specified on Day 0 and that I
deferred repeatedly.

---

## Day 4 (overnight) — a real leaderboard score, and the diagnosis I should have found on day 1

### The score

Two late submissions landed, both the same config, both scored identically:

| Submission | Public | Private |
|---|---|---|
| dummy `"A B C"` (day 1, for scale) | 0.375156 | 0.356882 |
| our retrieval + public reader | **0.761131** | **0.747994** |

Provenance, restated because it travels with the number: the corpus, chunking,
title-prefixing, BM25 index, query construction and offline pipeline are ours;
the reader is `mgoksu/llm-science-run-context-2`, a public checkpoint fine-tuned
by another competitor, permitted by the rules and cited in `CREDITS.md`. This is
not a model we trained.

**The pre-registered prediction was ~0.86 and it missed by 0.086.** The clean
gold 200 measured 0.8592 locally / 0.8600 on Kaggle, and the ~4,000-row hidden
test set came back at 0.7611. The gold set's own CI half-width is ±0.04, so
**the shortfall is larger than sampling noise on the holdout can explain.**
Two candidate causes, not separated: the 200-row official set may simply be an
easier sample, and/or the hidden test distribution differs from it. Logged as a
miss in `experiments/lb_log.csv` against the prediction rather than reframed
after the fact. The lesson is the one `PLAN.md`'s validation section already
states — a 200-row eval cannot resolve anything finer than ~4 points, and I
leaned on it for a point estimate anyway.

### The overnight runs falsified my own explanation

Two 5-hour runs, ~2,000 optimizer steps each (3.3× the 600 that the previous
attempt managed), everything else held fixed:

| Run | best MAP@3 | 95% CI |
|---|---|---|
| `night-large` (deberta-v3-large, frozen 18/24) | 0.3840 | [0.3649, 0.4036] |
| `night-base` (deberta-v3-base, frozen 6/12, maxlen 512) | 0.3746 | [0.3558, 0.3937] |

Neither clears the 0.3667 baseline. **Training volume was not the gap**, so the
"starved optimization" story from the previous entry is wrong too. That is the
third failed explanation for the same symptom.

### What it actually was: a train/eval source mismatch I introduced myself

Two lines of pandas, which were computable on day 1:

```
T1 dev (eval):      source 2 = 1,397/1,500 (93%),   source 1 = 103
39,249-row pool:    source 2 =   4,586     (11.7%), source 3 = 14,824 (38%)
```

cdeotte's `all_12_with_context2.csv` merges twelve generators whose question and
context styles differ. We trained on a pool that is 11.7% source-2 and evaluated
on a set that is 93% source-2 — a real covariate shift, not a cosmetic one.

**And this project had already found and fixed this once.** Row 5 of
`reports/ablation_table.md` — MAP@3 0.6086 [0.5880, 0.6291], still the best
own-model number here — was trained on a source-**matched** slice of exactly
4,586 rows, which is precisely source 2's count. When I scaled the pool
4,978 → 39,249 rows chasing "more data", I traded the distribution match for
volume and silently regressed a fix that was already in the log. Every reader
conclusion drawn after that point was measured on a broken setup.

### The pattern worth naming

Four explanations, in order: *reader never trained* (wrong — `ln(5)` loss means
uncalibrated, not unlearned), *failure is over-determined* (wrong — padding
around the real cause), *starved of optimizer steps* (wrong — falsified by 3.3×
the steps), *train/eval source mismatch* (supported, and previously known).

Each wrong version had confident arithmetic behind it. The failure was never
rigor within a step; it was **theorising about model behaviour before checking
that the training and evaluation data were drawn from the same distribution** —
and, twice, before re-reading what this project's own log already said. The two
measurements that actually discriminated (a known-good reader on our context;
a `value_counts()` on the source column) were both cheap and both available from
the start.

That is the finding I would lead with in an interview, ahead of any score.

### Running now

`ougridd/day4-src2-train`: the source-matched 4,586 rows, with **our**
general-corpus retrieval so train and eval retrieval match by construction,
frozen-layer recipe, 12 epochs ≈ 3,400 optimizer steps, 4h budget. It is the
clean test of the mismatch hypothesis. Beat targets: 0.3840 (ours, fails
baseline) and row 5's 0.6086 on cdeotte's context.

---

## Day 4 — the attribution table, and a pre-flight gate so the next hypothesis is cheap

Asked directly: what can be run locally so I stop burning GPU hours on wrong
hypotheses? The answer was an inversion I had available all project and never
made.

**We own a known-good reader.** `mgoksu/llm-science-run-context-2` scores 0.8600
on the clean gold 200 with our context. That makes it an **instrument for
testing our data**, not merely a score to envy: run it on a dataset and if it
scores well, the data is learnable and any failure is ours to fix by training;
if it scores badly, the data is the problem and no amount of reader training can
help. Every one of the three wrong hypotheses would have died in minutes against
that test.

`scripts/hypothesis_gate.py`, three checks cheapest-first:

| Check | Result | What it eliminates |
|---|---|---|
| 1. Row/context alignment (CPU, seconds) | **25/25 exact**, both the src2 training file and the T1 file | A silent row shift, which would wreck training while leaving inference-time retrieval correct -- indistinguishable from "the reader cannot learn" |
| 2. Train vs eval context quality (CPU, seconds) | train 0.6433 [0.5867,0.6967] vs eval 0.6133 [0.5567,0.6667], CIs overlap | "Our retrieval is worse on training data than eval data" |
| 3. Known-good reader on our eval + our context (GPU, ~5 min) | **0.7970 [0.7687,0.8247]** (n=500) | "Our context or eval set is inadequate" -- it is not |

### This is PLAN.md's attribution table, finally instantiated

The plan asked for an oracle-context ceiling on day 3 and it kept slipping
because there was no oracle. The known-good reader **is** the oracle:

```
known-good reader, T1 + our context      0.7970  [0.7687, 0.8247]
our trained reader, same data            0.3840  [0.3649, 0.4036]
-------------------------------------------------------------------
reader-attributable loss                 0.4130 MAP@3
```

With retrieval held fixed, ~41 points sit in reader training alone. That is an
exact figure rather than an impression, and it is the number that should have
driven every decision from day 2 onward.

It also retires a reframe I had floated an hour earlier -- "maybe T1 is simply a
harder yardstick than the official 200". Measured: 0.7970 on T1 vs 0.8600 on the
gold 200 with the same reader and the same retrieval. Modestly harder, not
broken. I am glad I checked rather than adopting it, because it was a
comfortable explanation that would have excused the gap.

### Decision recorded

The running source-matched job (`ougridd/day4-src2-train`) is **justified**: the
data supports ~0.80, so a properly-trained reader has real room. Explicitly not
a prediction that it reaches 0.80 -- the ceiling being there is not the same as
our recipe attaining it. Best own-model result to date is row 6's 0.6086
(source-matched, cdeotte's context); this run tests the same source matching
with our context.

And the standing rule, which is the actual deliverable of this entry: **no
further GPU run without the gate passing first.** Below 0.5 on check 3, the
quota does not get spent at all.

---

<!-- Append new entries above this line as work continues. -->
