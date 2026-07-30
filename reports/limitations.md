# Limitations

Grounded in what was actually found this session, not a generic checklist.

## ⚠ The single largest limitation, found on the last day: the reader, not retrieval, is the bottleneck

**The decisive measurement.** A public 2023 checkpoint
(`mgoksu/llm-science-run-context-2`, a leg of the notebook behind the published
0.823761) fed **our own** general-corpus BM25 top-5 context scores
**MAP@3 0.8592 [0.8200, 0.8958]** on the clean gold 200 (n=200, baseline
0.3667). Clean because that checkpoint's public training pools have zero prompt
overlap with the gold 200 (asserted in `scripts/build_context_train_pool.py`),
and a *lower* bound because it reads context from a retriever it was never
trained against.

**Therefore: the same context that yields ~0.43 with our reader yields 0.8592
with a well-trained one.** Corpus scope, BM25 ranking quality, per-option
retrieval, and the context-truncation budget are all second-order for this
project right now. Essentially all remaining score is in reader training —
layer freezing, LR schedule, training volume, and calibration.

**Secondary limitations of our reader training, each measured:**

1. **Starved optimization.** `lr=5e-6` with 10% warmup over 465 optimizer steps
   gives an effective LR of **1.63e-6** at optimizer step 15 — the step nearly
   every run selected as "best". Public 0.82–0.86 solutions used ~60k rows × 2
   epochs at 1e-5–3e-5.
2. **No layer freezing.** The 0.823761 notebook froze embeddings + the first 18
   of 24 layers (77.2M of 435.1M params trainable). Every run here trained all
   parameters — much less sample-efficient, and on a T4 at batch 2 a
   full-unfreeze `deberta-v3-large` completes only ~420 of 1,226 planned steps
   in a 75-minute budget.
3. **Poor calibration.** Training loss sat at ~`ln(5)` = 1.6094 in every run
   even where MAP@3 was strongly above baseline, meaning the reader's logits
   were nearly uniform. Harmless for MAP@3 in isolation (the metric is
   rank-based) but it blocks ensembling and confidence cascades, which need
   comparable probabilities.
4. **Context truncation, now known to be second-order.** At `max_length=384`
   the reader saw a median **26.8%** of its 5-chunk context (median 1,304
   tokens); full context fit for **1.0%** of rows (512 → 36.6%, 768 → 56.2%,
   1024 → 75.7%). Worth fixing, but the 0.8592 reference above used a
   comparable ~1,750-character budget, so this is not what separates us from
   0.86.

**A methodological limitation about this project's own reasoning, kept because
it is the most instructive thing here.** In one evening I asserted three
incompatible root causes for the same symptom — "the reader was never trained",
then "the failure is over-determined", then the correct
"miscalibrated-but-learning" — each with confident supporting arithmetic. Two
specific invalid inferences:

- **`ln(5)` loss was read as proof of no learning.** It is not: MAP@3 depends
  only on logit *rank order*, cross-entropy on their *magnitudes*, so a model
  can sit at the loss floor while ranking 25 SE above baseline (row 3 did
  exactly that: flat `ln(5)` loss, MAP@3 0.5641). `PLAN.md`'s own metric
  section states this property.
- **A 64-row overfit test was read as proof labels were aligned.** It is not:
  memorization succeeds even with scrambled labels. Labels were validated
  separately and do hold (longest-option-equals-answer at 32.7% train / 33.3%
  T1 vs 20% chance); that check also found **8.8% of training rows carry
  duplicate option text**, which is free label noise absent from T1.

The failure mode was reaching for a single-cause story and marshalling evidence
for it, instead of running the measurement that discriminates between
hypotheses. The one that finally did — a known-good reader on our own context —
was cheap, available from Day 1, and is precisely the "calibration anchor"
`PLAN.md` specified before any of this started.

## Measurement limitations

- **No source-article ground truth in any data tier.** Checked directly:
  `radek1`'s synthetic pool, `extra_train_set.csv`, and `t3_ood.parquet`
  all ship only prompt/options/answer, with no link back to a generating
  Wikipedia article. This means `PLAN.md`'s "Proxy A: source-article
  recall" retrieval metric cannot be computed as originally specified.
  `reports/retrieval_eval.md` uses an answer-support-recall proxy (Proxy B)
  instead, which brackets rather than measures true retrieval quality —
  see that report's own caveats on over/under-counting.
- **Best-checkpoint results are single-run, not cross-validated.** Every
  strong open-book/closed-book number in `reports/ablation_table.md`
  (0.5641, 0.6086) was selected as the best of many evaluations *within
  one training run*. This is standard early-stopping, not the
  best-of-N-*configs* cherry-picking `PLAN.md` warns against — but neither
  number has been confirmed with a second seed or on T3. Treat them as
  promising, not validated, operating points.
- ~~**The "learn fast, forget fast" dynamic recurred in every training run
  today** (closed-book, cdeotte source-matched, and — pending — the
  full-corpus own-retrieval run), always peaking within the first ~5-15%
  of a training run. This has not been root-caused; it's documented and
  worked around (frequent eval + best-checkpoint selection), not
  explained.~~
  **First withdrawn, then REINSTATED, both on 2026-07-30 (late Day 3).** I
  briefly withdrew this as "noise around a model pinned at the `ln(5)` loss
  floor". That withdrawal was wrong and is itself retracted: row 3's peak is
  MAP@3 0.5641 at n=1,500, ~25 standard errors above the 0.3667 baseline, which
  selection over a noisy metric cannot manufacture. The loss floor and a real
  peak coexist because MAP@3 reads only logit *rank order* while cross-entropy
  reads *magnitudes* — see the header of this file. **So the trajectory is
  real and remains un-root-caused**: still documented and worked around
  (frequent eval + best-checkpoint selection), not explained. The struck-through
  text above and this double reversal are both kept deliberately, since the
  sequence of the error is more instructive than a clean final answer.
- **No oracle-context ceiling or 2×2 failure decomposition yet.** Both
  need the full-corpus-trained reader's actual predictions, which weren't
  available at the time this file was written (full-corpus training was
  still running on Kaggle).
- **No hand-verification of the gold 200's labels.** `PLAN.md` calls for
  hand-checking the official 200 rows to establish a realistic label-noise
  ceiling; not done given the time budget. The gold set's evaluation
  budget (capped at ~8 uses per `CLAUDE.md`) has also not been spent at
  all yet — every number in this project so far is T1 or T3, never T2.

## Data and scope limitations

- **Single corpus, English only.** `mbanaei/all-paraphs-parsed-expanded`
  is the only retrieval corpus used; no second Wikipedia dump for
  diversity (`PLAN.md`'s workstream B), and non-English sources were never
  considered.
- **cdeotte's `60k-data-with-context-v2` has a real data-quality bug**
  (25.8% null-option rows, fixed) **and a real train/eval source
  mismatch** (its 12 constituent sources have different context
  characteristics). Both are documented, but the dataset was still used
  after fixing them — it wasn't discarded, and some results in this
  project's history (e.g. the 0.6086 source-matched run) depend on a
  4,586-row single-source slice of it, not the full 46k+ cleaned rows.
- **Answer-option length is a real, measured lexical shortcut** in this
  benchmark: a zero-parameter "longest option first" heuristic scores
  0.478 MAP@3 on T1, well above the 0.367 random baseline. Some fraction
  of every closed-book number in this project is very plausibly riding
  this shortcut rather than reflecting genuine science knowledge — not
  quantified separately from the "real" signal.
- **The MAP@3 metric has a 0.367 floor for 5-option questions.** Absolute
  numbers throughout this project (e.g. "0.61 MAP@3") look more impressive
  out of context than the actual margin over chance suggests; every number
  in `reports/ablation_table.md` is reported against that baseline
  explicitly for this reason.

## Compute and engineering limitations

- **Local GPU is memory-constrained (8 GB, effectively less under WSL2).**
  A WSL2-specific failure mode — silent overflow into slow system-RAM-backed
  "shared GPU memory," not a hard OOM — cost 5 hours before being caught
  and fixed with `src/llmsci/gpu_guard.py`'s memory cap + startup speed
  probe. Every local batch size in this project is chosen to fit safely
  under that constraint, not necessarily the throughput-optimal choice.
- **No dense retrieval, hybrid fusion, or reranking.** `PLAN.md` Day 3
  items 2–3 are explicitly deferred to a later session (see `PLAN.md`'s
  "today's compressed execution plan"), not attempted and found wanting.
- **No large model (`deberta-v3-large`, any 70B-class LLM) has been run.**
  The closed-book/open-book debugging this session consumed took priority;
  `PLAN.md`'s own arithmetic on 70B infeasibility under this project's
  compute budget was never re-verified empirically, only cited from the
  plan.
- **Kaggle GPU quota is being spent carefully but not tracked precisely.**
  The `get_accelerator_quota_statistics` API returned a malformed-looking
  duration string during this session and wasn't debugged further; total
  quota usage this week is not precisely known, only bounded by the
  handful of kernel runs logged in `DEVLOG.md`.
