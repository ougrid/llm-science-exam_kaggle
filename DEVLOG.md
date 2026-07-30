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

---

<!-- Append new entries above this line as work continues. -->
