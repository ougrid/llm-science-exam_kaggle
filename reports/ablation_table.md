# Ablation table (Day 1–3, in progress)

> ### Superseding correction (2026-07-31) — every own-model row below is invalid
>
> All own-model rows in this file were trained with **fp16 parameters**.
> `transformers` 5.x makes `from_pretrained` follow the checkpoint's stored
> dtype, and `deberta-v3-base`/`-large` both ship fp16, so every training run in
> this project — from the first — updated half-precision weights in place. At
> lr=2e-5 an AdamW update is ~1.3 ULP for a weight near 0.03 and **exactly zero
> for any weight ≥ 0.1**, which includes DeBERTa's LayerNorm weights. Controlled
> test, 16 rows / 60 steps / one variable: fp16 parks at loss 1.5687 (= ln 5),
> fp32 reaches 0.1013.
>
> So these numbers measure **a frozen encoder with a trainable head**, not the
> recipes their labels claim. They are kept, unedited, because the sequence of
> wrong diagnoses is the artifact. Read them as a floor, never as a result.
> Retrieval rows and public-checkpoint rows are unaffected — fp16 *inference* is
> fine, which is why the leaderboard score (0.761131) stands.
> Full story: `DEVLOG.md`, "the fifth hypothesis".

> **Second cause, found 2026-07-31 (supersedes the note above as a complete
> explanation).** Fixing the fp16 dtype did NOT make training work. The other
> cause is the learning rate, and it is an *interaction* with the layer freezing.
> Measured on the full 4,586-row pool, deberta-v3-base, 150 optimizer steps, one
> variable per cell:
>
> | | freeze 9/12 | freeze 0/12 |
> |---|---|---|
> | **lr 2e-5** | 1.6111 → 1.6117 null | 1.6112 → 1.5947 null |
> | **lr 1e-4** | 1.6116 → **1.4372 learns** | 1.6113 → 1.6129 null |
>
> Exactly one corner of four learns: the frozen lower layers are what make the
> higher LR usable, and the recipe inherited from cdeotte part 2 sat at the worst
> corner. `lr=2e-5` was used there on ~60k rows with 5.6× more trainable
> parameters and was never re-measured here.
>
> The two causes share one signature — `train_loss` pinned at ln(5) with healthy
> gradients — which is why fixing one produced no visible change. That is the
> generalisable trap, not either bug individually.




> **⚠ READ THIS FIRST — 2026-07-30, late Day 3.** Two corrections, in the
> order they happened, because the second retracts the first.
>
> **What I claimed first (RETRACTED):** that every run's training loss sitting
> at ~1.61 — with `ln(5) = 1.6094` being the loss of uniform predictions over
> 5 options — proved the reader never trained, making rows 3 and 5's peaks
> mere selection noise.
>
> **Why that was wrong:** row 3's closed-book run had flat `ln(5)` loss *and*
> MAP@3 **0.5641**, which at n=1,500 (per-row AP@3 SD ≈ 0.3, SE ≈ 0.008) is
> ~25 standard errors above the 0.3667 baseline. Both facts hold because
> **MAP@3 depends only on the rank order of the five logits while
> cross-entropy depends on their magnitudes** — a model with nearly-equal but
> consistently-ordered logits has loss ≈ `ln(5)` and still ranks well above
> chance. `PLAN.md`'s metric section says exactly this ("monotone rescaling of
> a *single* model's scores is a no-op for MAP@3"). So **`ln(5)` loss means
> badly calibrated, not unlearned**, and rows 3 and 5's peaks stand as real,
> as does the "learn fast, forget fast" trajectory.
>
> **What actually holds, and it is the project's most useful measurement:**
> a public 2023 checkpoint (`mgoksu/llm-science-run-context-2`) fed **our own**
> general-corpus BM25 top-5 context scores **0.8592 [0.8200, 0.8958]** on the
> clean gold 200 — a lower bound, since it reads context from a retriever it
> was never trained against. **Our retrieval is not the bottleneck; our reader
> is.** The same context yielding ~0.43 here yields 0.8592 with a well-trained
> reader, so corpus scope, BM25 ranking, per-option RRF, and the 26.8%
> truncation budget are all second-order for us: the remaining score lives
> almost entirely in reader training (layer freezing, LR schedule, training
> volume, calibration).
>
> **Still genuinely uncertain:** the reader-level verdicts below
> (phrase-match reranking −0.0490, context length 384→768 −0.0503, the
> general-corpus swap) were measured against a weak, poorly-calibrated reader.
> They are not refuted, but they are also not safe to generalize to a
> well-trained one — read them as "measured on a weak reader", not as
> properties of the techniques.
>
> Nothing below is deleted. The wrong numbers and the wrong conclusions stay
> visible with corrections attached, which is the point.

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
| 7 | open-book `deberta-v3-base`, our own BM25 retrieval (**full ~276k-article corpus**), best-checkpoint, *trained + originally scored on Kaggle* | ~~0.3869~~ **0.4297** | ~~[0.3681, 0.4058]~~ **[0.4099, 0.4496]** | Yes (re-scored) | **Corrected 2026-07-30 — see the environment-discrepancy note below.** Kaggle self-reported 0.3869 during training; re-evaluating the identical saved checkpoint locally gives 0.4297, meaningfully higher. The struck-through number is what training logged, kept visible rather than silently edited. |
| 8 | open-book `deberta-v3-base`, our own BM25 retrieval + **phrase-match rerank** (full corpus), best-checkpoint, *trained on Kaggle, scored locally* | 0.3807 | [0.3618, 0.3997] | Barely | Row 7's reranked counterpart, same correction applied (Kaggle self-reported 0.3947). **Paired bootstrap against row 7, both scored in the same (local) environment: −0.0490 [−0.0750, −0.0229] — reranking measurably HURT the trained reader**, the opposite of what Kaggle's own two separate numbers suggested. See below. |
| 9 | `reference_reproduction/`: **literal** `cdeotte/how-to-train-open-book-model-part-2` (own reimplementation, `deberta-v3-large`, 1,024-row demo subset, `NUM_TRAIN_SAMPLES` from the published notebook, final checkpoint) | 0.3770 | [0.3577, 0.3960] | No | A separate, clearly-labeled comparison track (see `reference_reproduction/RESULTS.md`), not part of this pipeline. The *published top solution's own literal configuration* doesn't clear baseline either — its 0.823761 requires "adjusting the parameters" per its own markdown. Trained and scored entirely locally, so unaffected by the row 7/8 correction. |

## Reference row — NOT this project's pipeline

Kept in its own table so it can never be misread as an own-pipeline result.
Using public models/datasets was explicit competition practice, so this is
legitimate; presenting it as ours would not be.

| # | config | MAP@3 | 95% CI | Tier | Notes |
|---|---|---|---|---|---|
| R1 | **PUBLIC CHECKPOINT** `mgoksu/llm-science-run-context-2` (a leg of the notebook behind the published 0.823761), fed **our own** general-corpus BM25 top-5 context, scored in its native part-2 inference format | **0.8592** | [0.8200, 0.8958] | **T2 — clean gold 200** | **The most informative measurement in the project.** Clean: that checkpoint's public training pools have zero prompt overlap with the gold 200 (asserted in `scripts/build_context_train_pool.py`). A *lower* bound: it reads context from a retriever it was never trained against. **Attribution: our retrieval is not the bottleneck — our reader is.** The identical context yields ~0.43 with our reader. Contrast the same weights on **contaminated** T1: 0.9170 [0.9056, 0.9282] — a number that must never be reported as a pipeline result. Cost one of the ~8 budgeted gold-set evaluations, logged with its reason. |

**Why R1 changes what to do next.** Every retrieval-side improvement
(corpus scope, BM25 ranking, per-option RRF, raising `max_length` past the
26.8% truncation budget) is second-order while the reader is the binding
constraint: a well-trained reader already extracts 0.8592 from the context we
produce today. The remaining score is in reader training — layer freezing
(the 0.82 notebook froze embeddings + 18/24 layers; we froze nothing), LR
schedule, training volume, and calibration.

## Reading the table honestly

- **Critical methodological finding, discovered late in Day 3: Kaggle and
  local evaluation of the identical saved checkpoint give different, and
  once inconsistent-in-direction, numbers.** Re-evaluating rows 7 and 9's
  checkpoints locally (same code, same data, same model weights) did not
  reproduce what Kaggle self-reported during training — and reversed the
  conclusion about whether reranking helped. Ruled out a script bug by
  re-running each checkpoint in complete isolation (separate processes);
  isolated numbers matched exactly, so this is a real environment effect,
  most likely a `transformers` version difference between Kaggle's docker
  image and this project's local venv (`transformers==5.14.1` locally;
  Kaggle's exact version wasn't captured and would need a dedicated check
  to confirm). **Consequence:** any Kaggle-trained checkpoint's score is
  only trustworthy if re-scored in the same environment as everything else
  it's being compared against. Rows 7 and 9 now show the locally-rescored
  numbers as primary. This also means each run's own **best-checkpoint
  selection** (done using Kaggle's environment during training) may not
  have picked the checkpoint that scores best locally — that can't be
  corrected retroactively without re-running training with local-environment
  scoring at each step, which wasn't done given the time budget. Flagged
  as an open gap, not silently absorbed into a clean-looking table.
- **With that correction applied, rows 6 vs 7 no longer look like "corpus
  coverage doesn't matter."** Row 6 (20k-article slice, always evaluated
  locally) is 0.3866 [0.3673, 0.4057]; row 7 corrected is 0.4297 [0.4099,
  0.4496] — CIs barely overlapping, not clearly the "statistically
  indistinguishable" result this table previously reported. That
  comparison itself now needs a same-environment rerun to trust either
  way; not done given the time budget. What **does** still hold, because
  it never depended on the disputed environment: the hand-inspection
  finding below (BM25 lacks phrase/entity awareness) was found by reading
  retrieved text directly, not by comparing scores across environments.
- **Rows 7 and 8 (phrase-match reranking), corrected: reranking hurt, not
  helped.** Kaggle's own two separate numbers (0.3869 → 0.3947) suggested
  an improvement; the proper same-environment paired bootstrap on the same
  two checkpoints shows −0.0490 [−0.0750, −0.0229], resolved in the other
  direction. The retrieval-level finding that motivated building the
  reranker still stands on its own (a pure metric over untrained retrieval,
  computed once, locally, never depending on a trained checkpoint):
  phrase-match reranking raised recall@5 from 0.510 to 0.585 on a 200-row
  T1 sample. What doesn't hold is the assumption that better retrieval
  ranking straightforwardly improves reader training — on this evidence,
  it did not, and the honest read is "not yet demonstrated," not "disproven
  by one confounded pair of runs."
- **Why the ranking diagnosis (BM25 lacks phrase/entity awareness) is still
  trustworthy despite the above**: it was found by hand-inspecting actual
  retrieved text for specific questions, not by comparing MAP@3 scores
  across environments. cdeotte's context (row 5) for a *Didymogenes*
  classification question opens with "Didymogenes is a genus of green
  algae in the class Trebouxiophyceae" — nearly verbatim the correct
  answer — while our own retrieval (rows 6, 7, 8 alike) returned
  *Ochromonas*, a different genus, even with the full corpus indexed. For
  a "Big Mama Thornton" question, ours retrieved a Chinese
  internet-censorship term and an unrelated athlete also named Thornton,
  fooled by matching "Big" + "mama" + "Thornton" as three independent
  tokens. That finding doesn't depend on any trained model's score, so the
  environment discrepancy above doesn't touch it — but it also means
  fixing the ranking problem (reranking) hasn't yet been shown to fix the
  downstream reader, which is a different, harder claim.
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
- **A separate comparison track reproduced a top public solution
  faithfully** (`reference_reproduction/`, row 9) rather than forking it
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

Last updated from `experiments/log.csv` through the 2026-07-30 Kaggle-vs-local
evaluation-environment correction (see `CRITICAL_diagnostic_kaggle_vs_local_eval_environment_discrepancy`
in `experiments/log.csv` for the full investigation).
