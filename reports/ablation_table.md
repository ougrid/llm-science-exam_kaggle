# Ablation table (Day 1–3, in progress)

All numbers are T1 (1,500-row synthetic dev set) MAP@3 with a 95% bootstrap
CI unless noted otherwise. Random baseline is the analytic value for 5
options: **0.3667**. Full per-run detail, including every invalid/superseded
run and the reasoning behind each fix, is in `experiments/log.csv` and
`DEVLOG.md` — this table is the condensed, presentable version, not a
replacement for that trail.

| # | config | T1 MAP@3 | 95% CI | Resolved vs. baseline? | Notes |
|---|---|---|---|---|---|
| 0 | random guess (analytic) | 0.3667 | — | — | Not `0.2` — a common error in public writeups. |
| 1 | options-only bias probe (zero-parameter length heuristic, no model) | 0.4780 | [0.4577, 0.4989] | Yes | A real lexical shortcut in this GPT-3.5-generated benchmark: correct answers run ~8% longer than distractors. Quantifying this was more valuable than hiding it. |
| 2 | closed-book `deberta-v3-base`, converged (end-of-training) | 0.3830 | [0.3640, 0.4021] | No | 3 epochs, `lr=5e-6, eps=1e-6`. Fine-tuning *destroys* the length-shortcut signal (row 1) without replacing it with anything. |
| 3 | closed-book `deberta-v3-base`, best-checkpoint-by-validation | **0.5641** | [0.5439, 0.5840] | Yes | Same run as row 2 — peaks at optimizer step 30 (within warmup), then collapses to row 2's floor. The "learn fast, forget fast" dynamic that recurred in every training run this project has produced so far. Caveat: likely partially rides the length shortcut (row 1), not purely closed-book knowledge — see `DEVLOG.md`. |
| 4 | mismatch row (paired): row-3 checkpoint fed our own BM25 top-5 context, **not retrained** | −0.0638 | [−0.0830, −0.0448] (paired delta, context − no-context) | Yes (context measurably *hurts*) | PLAN.md's predicted "flat or down" result, confirmed with a proper paired bootstrap after catching and fixing a sign-error bug in the first attempt. |
| 5 | open-book `deberta-v3-base`, cdeotte context, best-checkpoint, **train/eval source-matched** | **0.6086** | [0.5880, 0.6291] | Yes | 4,586 rows, one of cdeotte's 12 constituent sources, matched to T1's own dominant source. Same spike-then-collapse shape as row 3 (peak at step 15 of ~429). Best result in the project so far — but transient, and on a small single-source slice. |
| 6 | open-book `deberta-v3-base`, our own BM25 retrieval (20k-article slice), best-checkpoint, **fully self-consistent by construction** | 0.3866 | [0.3673, 0.4057] | Barely (CI lower bound clears baseline by 0.0006) | Same retriever for train and eval by construction — no source-mismatch possible — yet barely above baseline. Index covers only ~7% of the corpus. |
| 7 | open-book `deberta-v3-base`, our own BM25 retrieval (**full ~276k-article corpus**), best-checkpoint | 0.3869 | [0.3681, 0.4058] | Barely | Same recipe as row 6, 14x the corpus coverage, **statistically indistinguishable result**. This corpus's retrieval eval harness (`reports/retrieval_eval.md`) measures real recall@5=0.481 — so recall existing in aggregate did not translate into training gains. See below: hand-inspection found why. |
| 8 | `reference_reproduction/`: **literal** `cdeotte/how-to-train-open-book-model-part-2` (own reimplementation, `deberta-v3-large`, 1,024-row demo subset, `NUM_TRAIN_SAMPLES` from the published notebook, final checkpoint) | 0.3770 | [0.3577, 0.3960] | No | A separate, clearly-labeled comparison track (see `reference_reproduction/RESULTS.md`), not part of this pipeline. The *published top solution's own literal configuration* doesn't clear baseline either — its 0.823761 requires "adjusting the parameters" per its own markdown. Statistically indistinguishable from row 7's final-checkpoint result on different context. Corroborates that the 0.37→0.61 gap (rows 7 vs 5) is checkpoint-selection, not architecture or model scale. |

## Reading the table honestly

- **Rows 5, 6, and 7 are the money comparison, and the story is more
  interesting than "recall explains it."** Rows 6→7 rule out corpus
  coverage as the lever: 14x more articles indexed, same flat result. What
  actually explains the ~0.22 gap to row 5 was found by hand-inspecting
  the retrieved text itself, not by another metric: cdeotte's context
  (row 5) for a *Didymogenes* classification question opens with
  "Didymogenes is a genus of green algae in the class Trebouxiophyceae"
  — nearly verbatim the correct answer — while our own retrieval (rows 6
  and 7 alike) returned *Ochromonas*, a different genus, even with the
  full corpus indexed. For a "Big Mama Thornton" question, ours retrieved
  a Chinese internet-censorship term and an unrelated athlete also named
  Thornton, fooled by matching "Big" + "mama" + "Thornton" as three
  independent tokens. **BM25's bag-of-words scoring has no phrase- or
  entity-level understanding, and that failure mode doesn't improve with
  more corpus coverage** — the problem is ranking what's already
  retrievable, not whether the right article exists in the index. Full
  writeup in `DEVLOG.md`'s "the finding that actually explains the gap."
- **Row 3 and row 5's peaks are both best-of-N-evaluations selections**
  within their own training run (checkpoints saved whenever validation
  MAP@3 improved). This is standard early-stopping practice
  (`Trainer(load_best_model_at_end=True)`'s default behavior), not the
  best-of-N-*configs* cherry-picking `PLAN.md` warns against — but the
  optimism-correction arithmetic in `PLAN.md`'s validation section still
  applies in miniature, and neither number should be treated as a stable,
  reproducible operating point without further validation (e.g. a second
  run with a different seed, or confirmation on T3).
- **PLAN.md's Day 2 target was ~0.82–0.86.** Not reached at pilot scale.
  The gap is not unexplained: it traces to (a) a real data-quality bug
  (25.8% null-option rows in the external context dataset), (b) a real
  train/eval retriever-source mismatch, and (c) our own retriever's lack
  of phrase/entity-level ranking — not, as first suspected, its recall.
  None of these are dead ends. The next lever identified (phrase-aware or
  entity-aware retrieval, e.g. a reranker) is unexhausted, not ruled out.
- **Recall@k, MRR, and nDCG for the full corpus are measured** in
  `reports/retrieval_eval.md` (T1 recall@5=0.481, recall@100=0.711;
  T3-OOD recall@100=0.588, a real measured distribution shift). The
  oracle-context ceiling and 2×2 failure decomposition still need design
  work given the lack of source-article ground truth (see that report).
- **A cheap phrase-match rerank (no new model) improved recall@5 from
  0.510 to 0.585** on a 200-row T1 sample (`src/llmsci/retrieve/rerank.py`)
  — confirming the ranking diagnosis above. A retrain against this
  reranked context is in flight; result pending in `experiments/log.csv`.
- **A separate comparison track reproduced a top public solution
  faithfully** (`reference_reproduction/`, row 8) rather than forking it
  — the legitimate-reuse process `PLAN.md` calls for, elevated to a full
  parallel track given time pressure. Its most valuable output wasn't the
  score: it found that **PLAN.md's "reproduce a notebook for a calibration
  anchor" strategy doesn't work for this competition** (the shipped
  checkpoint scores 0.9170 on T1 but is 100% contaminated — trained on
  the same public pools T1 was built from), which is now struck from
  `PLAN.md`. Its reranker experiment also independently found the same
  small-but-real improvement pattern this pipeline found (+0.0153
  [+0.0007, +0.0293] in-window recall on their pipeline), and reported
  that it declined to run the expensive reader-level MAP@3 delta because
  their eval couldn't resolve it — a good model for when *not* to spend a
  training run.
- Not yet run: dense retrieval, hybrid RRF fusion — explicitly deferred
  per `PLAN.md`'s time-budget decision, not attempted and found wanting.

Last updated from `experiments/log.csv` through the 2026-07-30 phrase-match
rerank test and the `reference_reproduction/` comparison track.
