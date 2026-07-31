# LLM Science Exam — a reproduction study of retrieval vs. scale

A 4-day reproduction of Kaggle's closed [LLM Science Exam](https://www.kaggle.com/competitions/kaggle-llm-science-exam)
competition (5-option science MCQ, MAP@3, internet-disabled offline
inference). The competition's headline finding is a judgment result, not a
trick: a 435M-parameter encoder with good retrieval reached top-5 while 3rd
and 5th place needed 70B models to do marginally better.

**What this repo is actually for:** measuring *where the score comes from*,
with honest statistics — not chasing a leaderboard number. The most useful
output turned out to be an attribution result, and a three-day debugging trail
that falsified four of my own successive hypotheses before measurement found the
two real causes.

---

## Headline

> **Kaggle leaderboard: 0.762796 public / 0.755183 private** — my retrieval
> pipeline paired with a public reader checkpoint (provenance below), against
> 0.375156 for a random-guess baseline.
>
> **The finding that matters more than the score:** holding the retrieved
> context *fixed* and swapping only the reader moves MAP@3 from **0.3840**
> (mine) to **0.8600** (a public 2023 checkpoint) on the same clean 200-row
> holdout. The retrieval was never the bottleneck — the reader *training* was.

That comparison is the project: it localises the failure to one component by
changing exactly one thing. What it cost to find the cause is the rest of it.

**Six hypotheses, four wrong.** Every training run in this repo pinned
`train_loss` at exactly `ln(5) = 1.6094` — the cross-entropy of predicting
uniformly — while gradients, the optimizer, the labels, the retrieved context and
the input format were all measurably healthy. I blamed the data four times in a
row. The real causes were **two borrowed constants**, and they shared one
signature, which is why fixing the first looked like a failure:

| | cause | how it hid |
|---|---|---|
| 1 | **fp16 parameters** — `transformers` 5.x follows the *checkpoint's* dtype, and `deberta-v3` ships fp16, so AdamW updated half-precision weights in place. At lr=2e-5 an update is ~1.3 ULP near a weight of 0.03 and **exactly zero for any weight ≥ 0.1**. | `transformers>=4.46`, unpinned |
| 2 | **lr=2e-5 × freeze 18/24** — inherited from a public notebook that used it on ~60k rows with 5.6× more trainable parameters. A 2×2 showed **exactly one corner of four learns**, because the frozen lower layers are what make a higher LR usable. | `LR = 2e-5  # cdeotte part 2's value` |

The second is the one I'd want to be asked about. A bare magic number invites
scrutiny; the same number with a provenance comment reads as *already justified*,
so it looked like an answer rather than a question every time I re-read the file.
Full narrative in `DEVLOG.md`; every measurement in `experiments/log.csv`.

## Results

Random baseline is the analytic **0.3667** for 5 options at k=3 — not 0.2, a
common error in public writeups. Every number carries a 95% bootstrap CI.

### Mine

| # | What | MAP@3 | 95% CI | Clears baseline? |
|---|---|---|---|---|
| 0 | random guess (analytic) | 0.3667 | — | — |
| 1 | options-only bias probe (zero parameters, no model) | 0.4780 | [0.4577, 0.4989] | Yes |
| 2 | closed-book `deberta-v3-base`, best checkpoint | 0.5641 | [0.5439, 0.5840] | Yes |
| 3 | open-book, my BM25 + general Wikipedia, `deberta-v3-large`, frozen-layer recipe, 600 steps | 0.3807 | [0.3618, 0.3997] | **No** |
| 4 | same, **3.3× the training steps** (~2,000, 5 h) | 0.3840 | [0.3649, 0.4036] | **No** |
| 5 | same on `deberta-v3-base`, `max_length=512`, 2 epochs | 0.3746 | [0.3558, 0.3937] | **No** |
| 6 | **source-matched** slice (4,586 rows), cdeotte's context — best of the invalid runs | 0.6086 | [0.5880, 0.6291] | Yes |

> ⚠️ **Rows 2–6 are invalid as measurements of their stated recipes.** Every one
> trained fp16 *parameters* at a learning rate later measured to be incapable of
> learning. They record a frozen encoder with a partly-trainable head — the small
> head weights are the one magnitude regime where fp16 still moves — so read them
> as floors, not results. Rows 0, 1 and the retrieval numbers are unaffected, as
> is the leaderboard score (fp16 *inference* is correct and cheaper).

Rows 3→5 are still informative as a *falsification*: **tripling the optimizer
steps changed nothing** (0.3807 → 0.3840), which killed "undertrained" as the
explanation. I then attributed row 6's 0.6086 to source-matched training data —
that was hypothesis 4, and it is also wrong. A known-good reader scores 0.8037 on
my *training* file versus 0.7970 on the eval file, so the two are equally readable
and generator shift was never a mechanism. Row 6 was simply the run whose
conditions let the head learn most before fp16 rounding stalled it.

### On the Kaggle leaderboard

| Submission | Public | Private | Predicted beforehand |
|---|---|---|---|
| dummy `"A B C"` (scale reference) | 0.375156 | 0.356882 | ~0.3667 ✓ |
| single-index retrieval + public reader | 0.761131 | 0.747994 | ~0.86 ✗ **missed by 0.086** |
| **dual-index RRF fusion + public reader** | **0.762796** | **0.755183** | 0.770 (range 0.755–0.785) — inside range, **0.007 high** |

The third row is the one measurement-driven score improvement in this project, and
the prediction attached to it is the more useful artifact. **+0.0072 private,
+0.0017 public.** I predicted +0.009 and the *private* split — the larger, less
noisy one — moved +0.0072, so the effect-size reasoning was right; I quoted it
against the noisier number. The calibration improvement over the previous row
(0.007 vs 0.086) came from one change: anchoring the prediction on the **actual
leaderboard score** plus a measured delta, instead of on a 200-row local estimate.
Both the revert trigger (below 0.761) and the escalate trigger (above 0.785) were
written into `experiments/lb_log.csv` before submitting; neither fired.

On the 0.086 miss in row two: the 200-row holdout said 0.86, the ~4,000-row hidden
test said 0.76, and that gap exceeds the holdout's own ±0.04 CI — so it is **not**
explainable by sampling noise alone. Either the official 200 is an easier sample or
its distribution differs from the hidden set; one submission cannot separate those.
What it did establish is that local gold-200 figures are the wrong anchor for a
leaderboard prediction, which is exactly the correction that made row three land.

Row 1 is worth dwelling on: a **zero-parameter heuristic that always picks the
longest option scores 0.4780**, beating several trained configurations. This
GPT-3.5-generated benchmark has a real lexical shortcut, and quantifying my own
benchmark's artifact mattered more than any single score.

Row 3 is the uncomfortable one: my best-engineered pipeline is **worse than
row 2's closed-book model and worse than row 1's heuristic.**

### Valid own-model results (both faults fixed)

The recipe below is the first in this project where every constant traces to a
measurement in `experiments/log.csv` rather than to a source notebook:
`dtype=torch.float32` + fp16 autocast, **lr=1e-4**, freeze embeddings + 75% of
layers, effective batch 16, maxlen 384.

| What | MAP@3 (full 1,500-row T1) | Notes |
|---|---|---|
| random baseline | 0.3667 | analytic |
| **`deberta-v3-base`, measured recipe, 4,586 rows** | **0.6906** [0.6714, 0.7090] | best checkpoint (step 1200 of 2288), rescored on all 1,500 rows |
| `deberta-v3-large`, **same recipe** | 0.3639 [0.3217, 0.4072] | **does not train** — 282 steps at full LR, never leaves ln(5) |
| known-good public reader (ceiling) | 0.7793 [0.7628, 0.7958] | same eval set, same retrieval |

Both numbers above are on the **same 1,500 rows**, which matters: the 0.7970 ceiling
this README quoted for most of the project came from a 500-row sample and was
inflated by 0.0177. Selection subsets on this eval set drift 0.018–0.032 in *either*
direction — my reader measured 0.6693 on a 500-row sample versus 0.6906 on all
1,500 — so any figure derived by subtracting across different samples compounds two
errors instead of cancelling them.

**The 435M model does not train under this compute budget and the 184M one
does.** That is measured, not assumed: identical pool, identical recipe, identical
freezing fraction (18/24 = 9/12 = 75%), and large had 282 steps at full learning
rate while base broke away within ~60. `PLAN.md` predicted `deberta-v3-large`
would be the workhorse — it reached top-5 in the real competition — and the
measurement says otherwise for *my* budget. Building the eval harness before the
training runs is what made that visible instead of a mystery.

So `deberta-v3-base` is the headline reader. It sharpens the project's thesis
rather than conceding it: the original claim was *retrieval quality beats
parameter count*, and this is a harder version — a 184M reader that trains beats
a 435M one that doesn't.

**The boundary on that claim, stated so it isn't over-read.** What is measured is
that `deberta-v3-large` does not train *at three learning rates (2e-5, 5e-5,
1e-4), with 75% of layers frozen, at effective batch 16, on a 4,586-row pool*.
That is a bounded negative result about a configuration, **not** evidence the
architecture is untrainable — cdeotte trained the same checkpoint successfully on
~60k rows, and the most likely difference is data scale, not the model. The
correct follow-up is to retry large on the 39,249-row pool before drawing any
architectural conclusion, and that is queued rather than claimed.

### Not mine (reference anchor — separated deliberately)

| # | What | MAP@3 | 95% CI | Tier |
|---|---|---|---|---|
| R1 | **public** `mgoksu/llm-science-run-context-2` + **my** retrieval | **0.8592** | [0.8200, 0.8958] | clean gold 200 (local) |
| R1′ | same, re-run inside a Kaggle kernel | **0.8600** | [0.8208, 0.8967] | clean gold 200 (Kaggle) |

R1 vs R1′ is also a useful cross-environment check: they agree closely, unlike
this project's earlier finding that *trained* checkpoints score differently on
Kaggle vs. locally. The inference path reproduces; the training-time scoring
did not.

R1 is clean (that checkpoint's public training pools have zero prompt overlap
with the official 200, asserted in `scripts/build_context_train_pool.py`) and is
a **lower bound**, since it reads context from a retriever it was never trained
against. The same weights score 0.9170 on my T1 dev set — a number that is
**100% contaminated** and appears here only as a warning, never as a result.

## The attribution table

`PLAN.md` asked for an oracle-context ceiling to split reader loss from
retrieval loss. There was no oracle available — until it became clear that the
known-good public reader *is* one. Hold retrieval fixed, vary only the reader:

| Reader, on the **same 1,500 T1 rows** with **my** retrieved context | MAP@3 | 95% CI |
|---|---|---|
| known-good public checkpoint | 0.7793 | [0.7628, 0.7958] |
| **mine** (`deberta-v3-base`, measured recipe) | **0.6906** | [0.6714, 0.7090] |
| **reader-attributable loss** (paired bootstrap) | **+0.0888** | **[+0.0707, +0.1063]** |

So ~9 MAP@3 points sit in reader training, and everything above 0.7793 is what
better retrieval could add. The CI excludes 0, so it is resolved.

**This supersedes a figure of 0.4130 that this README carried for most of the
project, and the correction is the most instructive thing here.** That number was
wrong twice over:

1. **It quantified bugs, not a training gap.** The reader it measured was training
   fp16 *parameters* at an inherited learning rate that measurement later showed
   cannot learn at all. **79% of the 0.4130 was those two bugs.**
2. **It subtracted across different row samples** — a 500-row seed-0 sample for the
   ceiling against other rows for my reader. Measured drift from sample choice on
   this very eval set: my reader 0.6693 (500 rows, seed 42) vs 0.6907 (all 1,500);
   the ceiling 0.7970 (500 rows, seed 0) vs 0.7793 (all 1,500). The two drifts ran
   in **opposite** directions, so the cross-sample subtraction compounded both
   errors instead of cancelling them.

`PLAN.md`'s own noise arithmetic predicts exactly this: ±0.032 on 500 rows, ±0.018
on 1,500. I wrote that section before building anything and then made a ±0.03-sized
error anyway. The fix is method, not care — `scripts/attribution_paired_full_t1.py`
scores both readers on identical rows and bootstraps the **per-row difference**,
which is why its CI is ±0.018 rather than the much looser interval a subtraction of
two independent means would give.

The same reader scores 0.8600 on the official 200 with the same retrieval, so
T1 is *modestly* harder than the gold set — not a broken eval. I had floated
"maybe T1 is just a harder yardstick" as an explanation for the gap an hour
before measuring it; it was a comfortable story and it was mostly wrong.

### The pre-flight gate (`scripts/hypothesis_gate.py`)

The generalizable lesson from six hypotheses, four of them wrong. The move is an inversion:
**use a known-good model as an instrument to test your data, not as a score to
chase.** Scores well → the data is learnable, the failure is yours to fix by
training. Scores badly → the data is the problem and no training run can help.

| Check | Cost | Result |
|---|---|---|
| Trainable parameter dtype | **ms** | **fp16 — untrainable** (added after the gate failed; see below) |
| Row/context alignment | seconds | 25/25 exact — no silent row shift |
| Train vs eval context quality | seconds | 0.6433 vs 0.6133, CIs overlap — comparable |
| Known-good reader on target eval | ~5 min | 0.7970 — data is learnable |

Standing rule: **no GPU run without the gate passing first.**

### The gate passed, and the next run still failed

Worth keeping, because it is the most useful thing in this repo. All three
original checks passed in full — and the next GPU run pinned at ln(5) anyway.
Every check interrogated the *data*, because all four of my hypotheses had blamed
the data. Nothing looked at the model's own parameters.

The cause was one library default. `transformers` 5.x makes `from_pretrained`
follow the **checkpoint's** stored dtype, and `deberta-v3-base`/`-large` both
ship fp16, so the bare call returns fp16 **parameters** — not mixed precision,
but half-precision weights AdamW updates in place. Under 4.x the same line gave
fp32; `pyproject.toml` said `transformers>=4.46` with no upper bound.

Measured, 20 AdamW steps at lr=2e-5 with a constant-sign gradient:

| weight magnitude | fp16 movement | fp32 movement |
|---|---|---|
| 0.03 | 3.05e-04 (76% of intended) | 4.00e-04 |
| **≥ 0.10** | **0.00e+00 — frozen** | 4.00e-04 |

fp16 ULP grows with magnitude, so larger weights freeze completely. DeBERTa's
LayerNorm weights sit near 1.0. One step on the real recipe moved
`classifier.weight` by `1.072e-01` in fp16 versus a correct `2.001e-05` in fp32.

It is retrodictive, which is why I trust it: it explains ln(5) on both a T4 and a
Blackwell card (a library default, not hardware), why 3.3× more steps changed
nothing, why source-matching changed nothing, why *inference* was always fine
(0.761131 on the real leaderboard — fp16 forward passes are fine), and why the
0.5641 closed-book peak decayed and row 6 topped out at 0.6086: the small
head weights are the one regime where fp16 does move, so the head learned while
the encoder stayed frozen.

Three things I would want an interviewer to take from this:

1. **`train_loss`, not eval MAP@3, was the tell from run #1.** 77.2M trainable
   parameters cannot fail to reduce *training* loss on 4,586 rows over four
   epochs. Every run printed it. I read it four times as "the data is wrong."
2. **Four hypotheses, four wrong, all in the same category.** Each had arithmetic
   behind it; none was a measurement that could come back *negative*. The check
   that finally cleared the data — a known-good reader scored on my *training*
   file — took five minutes with an instrument already built.
3. **The fix is 3 s of tests** (`tests/test_trainable_dtype.py`, asserting the
   mechanism: frozen at w ≥ 0.1, ULP-quantized when it moves, ln(5) = uniform-
   prediction loss). The check that would have saved four GPU runs was always
   this cheap. Full narrative in `DEVLOG.md`, "the fifth hypothesis".

## Retrieval, measured independently of the reader

No source-article ground truth exists in any tier (checked directly), so
`PLAN.md`'s article-recall proxy isn't computable. This uses answer-support
recall (the DPR paper's answer-string protocol), which brackets rather than
measures truth.

| Change | Effect on recall@5 | Resolved? |
|---|---|---|
| STEM-only corpus → general Wikipedia | **+0.1393** [0.1167, 0.1620] | Yes |
| pooled query → per-option retrieval + RRF | **−0.0780** [−0.0960, −0.0607] | Yes (it *hurt*) |

The RRF result is a genuine negative: `PLAN.md` called per-option retrieval
"the single highest-expected-value trick", and it lost by 8 points at 4× the
query cost. Its own reasoning explains why — my pooled query already contains
all five option texts, which is a free HyDE, so per-option retrieval adds
dilution rather than coverage.

Also measured: at `max_length=384` the reader sees a **median 26.8%** of its
retrieved context (median 1,304 tokens), and the full context fits for **1.0%**
of rows. So effective retrieval quality was nearer recall@1 (0.43) than
recall@5 (0.62). Second-order given R1, but real.

## Where the score actually went — and four explanations, three of them wrong

The debugging trail is the honest value here, so `DEVLOG.md` keeps it in full,
including the wrong turns in the order I took them.

| # | My explanation | Verdict |
|---|---|---|
| 1 | "The reader never trained — loss is pinned at `ln(5)`" | **Wrong** |
| 2 | "The failure is over-determined (LR + freezing + truncation)" | **Wrong** — padding around the real cause |
| 3 | "It's starved of optimizer steps" | **Wrong** — falsified by 3.3× the steps |
| 4 | "Train and eval data come from different generators" | **Supported** — and already in my own log |

**Why (1) was wrong, and it's a metric subtlety worth knowing.** `ln(5) = 1.6094`
is the cross-entropy of uniform predictions over 5 options, and every run sat
there. But MAP@3 depends only on the *rank order* of logits while cross-entropy
depends on their *magnitudes*, so a model with nearly-equal but consistently
ordered logits sits at the loss floor and still ranks well above chance. The
closed-book run proves it: flat `ln(5)` loss *and* MAP@3 0.5641, which is ~25
standard errors above baseline at n=1,500. **`ln(5)` means uncalibrated, not
unlearned.** This document's own metric notes state the property; I failed to
apply it.

**Why (3) was wrong.** Two 5-hour runs at ~2,000 optimizer steps moved 0.3807 →
0.3840. Volume was not the constraint.

**What (4) actually is** — two lines of pandas, computable on day 1:

```
T1 dev (eval):     source 2 = 1,397/1,500 (93%),   source 1 = 103
39,249-row pool:   source 2 =   4,586     (11.7%), source 3 = 14,824 (38%)
```

`cdeotte/60k-data-with-context-v2` merges twelve generators with different
question and context styles. I trained on 11.7% source-2 data and evaluated on
93% source-2 data. **And this repo had already fixed that once:** row 6's 0.6086
was trained on a source-matched slice of exactly 4,586 rows — precisely source
2's count. Scaling the pool 4,978 → 39,249 rows for "more data" traded the
distribution match for volume and regressed a known fix. Every reader conclusion
after that point was measured on a broken setup.

**A second invalid inference, for completeness.** I ran a 64-row overfit test,
drove loss to 0.0018, and claimed it proved labels were aligned. It doesn't —
memorization succeeds with scrambled labels. The valid check was the benchmark's
length shortcut, which holds (32.7% train / 33.3% dev vs 20% chance) and which
also surfaced **8.8% of training rows carrying duplicate option text**.

**The pattern.** Every wrong version had confident arithmetic behind it. The
failure was never rigor inside a step; it was theorising about model behaviour
before checking that train and eval data were drawn from the same distribution,
and twice before re-reading what this project's own log already said. Both
measurements that finally discriminated — a known-good reader on my context, and
a `value_counts()` on a source column — were cheap and available from the start.

## Validation design

The part I would defend hardest, and mostly the part that held up.

- **The official 200 rows were never trained on.** They are the held-out test
  set, and the project spent **1 of ~8** budgeted evaluations on them (R1),
  logged with a date and reason.
- **Noise arithmetic before conclusions.** Per-row AP@3 ∈ {1, ½, ⅓, 0} with
  SD ≈ 0.3, so a 200-row eval has a CI half-width of ±0.04 and **cannot
  resolve anything smaller than a ~4-point change**. Stated before results, not
  after.
- **Paired bootstrap for every comparison**, never two separate CIs — 10,000
  resamples on per-row differences. This caught a sign-error that had inverted
  one conclusion, and it reversed the reranking result once both checkpoints
  were scored in the same environment.
- **Environment discrepancy found and documented:** identical checkpoints score
  differently on Kaggle vs. locally (0.3869 vs 0.4297 for the same weights),
  once in the opposite *direction*. Ruled out a script bug by re-running each in
  isolation. Every Kaggle-scored number is therefore only comparable to other
  Kaggle-scored numbers.
- **Article-level, not row-level, leakage checks**, plus a contamination audit
  that found the public checkpoint 100% contaminated against T1 — which is why
  R1 uses the gold 200 instead.

## What I'd do next, from the attribution rather than from vibes

R1 says a well-trained reader extracts 0.86 from the context I already produce.
So retrieval work is second-order and the ranked list is:

1. **Train the reader properly.** The 0.82 notebook froze embeddings + 18 of 24
   layers (77.2M of 435.1M trainable) and ran 2 full epochs; my best run managed
   600 of 2,453 steps inside its time budget and never converged. This is the
   whole gap.
2. **Fix calibration** — needed for ensembling and confidence cascades, both
   currently blocked by near-uniform logits.
3. **Then** raise `max_length` past the 26.8% truncation budget and index more
   than 1.6M of the 21.6M available chunks.
4. **Not** a larger model: R1 is `deberta-v3-large`-class, and it already
   clears 0.85 on my context.

## Repo map

```
PLAN.md            technical plan; "Final-day score strategy" has the ranked plan
DEVLOG.md          the honest narrative, including every wrong turn
CREDITS.md         what was taken from whom, and what is mine
experiments/log.csv every run: config, git SHA, CI, hypothesis, verdict
reports/           ablation_table.md · retrieval_eval.md · limitations.md
src/llmsci/        metrics · corpus · retrieve/{sparse,rerank,fuse,eval} · reader/mc
scripts/           build/train/eval entry points, incl. the diagnostics above
notebooks/kaggle/  offline submission + training kernels actually run
tests/             52 tests: metrics, collator shapes, leakage, submission format
```

## Honest limitations

Full list in `reports/limitations.md`. The ones that matter most: my own trained
reader never cleared baseline in the time available; the strong number (R1)
depends on someone else's checkpoint; the answer-support recall proxy brackets
rather than measures retrieval truth; and the benchmark's own labels are
GPT-3.5-generated, so its length shortcut (row 1) is baked into every
closed-book number here.
