# Limitations

Grounded in what was actually found this session, not a generic checklist.

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
- **The "learn fast, forget fast" dynamic recurred in every training run
  today** (closed-book, cdeotte source-matched, and — pending — the
  full-corpus own-retrieval run), always peaking within the first ~5-15%
  of a training run. This has not been root-caused; it's documented and
  worked around (frequent eval + best-checkpoint selection), not
  explained.
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
