# Ablation table (Day 1–2, in progress)

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
| 6 | open-book `deberta-v3-base`, our own BM25 retrieval, best-checkpoint, **fully self-consistent by construction** | 0.3866 | [0.3673, 0.4057] | Barely (CI lower bound clears baseline by 0.0006) | Same retriever for train and eval by construction — no source-mismatch possible — yet barely above baseline. Index covers only ~7% of the corpus (20k of ~276k articles), so most T1 source articles aren't retrievable at all. **Shows retriever consistency alone isn't sufficient — recall/quality is the dominant lever.** |

## Reading the table honestly

- **Rows 5 vs 6 are the money comparison.** Same model, same recipe,
  same best-checkpoint methodology — the only real difference is retrieval
  *quality* (an external pipeline of unknown but evidently higher recall,
  vs. our own BM25 over a small corpus slice). The ~0.22 gap between them
  is the cleanest evidence in this project that recall dominates over
  architecture or training recipe, which is the whole thesis this
  reproduction study set out to test.
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
  train/eval retriever-source mismatch, and (c) evidence that our own
  retrieval's low recall (small corpus slice) caps the achievable signal
  regardless of training recipe. None of these are dead ends — the next
  lever (full-corpus retrieval, more training data/epochs) is identified
  and unexhausted, not ruled out.
- Rows not yet run: dense retrieval, hybrid RRF, reranking, the oracle-context
  ceiling, and the full recall@k retrieval-eval harness (Day 3 work).

Last updated from `experiments/log.csv` through the 2026-07-30 Kaggle run
(`deberta-v3-base_open-book_OWN-RETRIEVAL_..._KAGGLE-T4x2`).
