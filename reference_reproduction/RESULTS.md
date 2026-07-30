# Reference reproduction: cdeotte's open-book DeBERTa pipeline

Comparison track, run 2026-07-30. **Not part of the main pipeline** — nothing here feeds
`src/llmsci/`, `experiments/log.csv`, `reports/`, or the headline ablation table, and no
gold-set evaluation was spent on it. Its only job is to put a known-good 2023 public
solution on the same ruler as this project's own numbers.

Architectural write-up of the target, written before any code here: `NOTES.md`.
Unmodified pulled notebooks: `original/`.

## What was reproduced, and from where

| | |
|---|---|
| Target | `cdeotte/how-to-train-open-book-model-part-2` — public LB **0.823761** |
| URL | https://www.kaggle.com/code/cdeotte/how-to-train-open-book-model-part-2 |
| Also pulled | `cdeotte/how-to-train-open-book-model-part-1` (the training half part 2 consumes) |
| URL | https://www.kaggle.com/code/cdeotte/how-to-train-open-book-model |
| Retrieved | `kaggle kernels pull`, 2026-07-30, raw `.ipynb` kept in `original/` |
| Reader reproduced | `microsoft/deberta-v3-large` `AutoModelForMultipleChoice`, embeddings + first 18 of 24 layers frozen, `MAX_INPUT=256`, `truncation='only_first'`, input `[CLS] ctx #### prompt [SEP] option [SEP]`, lr 2e-5, cosine, warmup 0.1, wd 0.01, effective batch 16, 2 epochs, final checkpoint (no selection) |
| Retrieval | **not** re-run — see "What I could not reproduce" |
| Eval set | `data/t1_dev.csv`, n=1500, this project's own T1 dev tier. Gold 200 untouched. |

Reimplemented from the prose in `NOTES.md` into `reproduced/`, not copy-pasted. The one
piece deliberately reused rather than rewritten is `llmsci.metrics` — CLAUDE.md requires a
single source of truth for MAP@3 and bootstrap CIs, and reusing it is what makes these
numbers comparable to the main pipeline's at all.

## Headline numbers

All on the same 1,500 T1 dev rows, 95% percentile bootstrap CIs, 10,000 resamples.
Analytic random baseline **0.3667**.

| # | Configuration | T1 MAP@3 | 95% CI | Resolved vs random? |
|---|---|---|---|---|
| 1 | Random guess (analytic) | 0.3667 | — | — |
| 2 | **Faithful reproduction, literal notebook** (1,024 train rows) | **0.3770** | [0.3577, 0.3960] | **No — CI contains 0.3667** |
| 3 | "Longest option first" heuristic (main pipeline, for scale) | 0.4780 | [0.4577, 0.4989] | Yes |
| 4 | `mgoksu/llm-science-run-context-2`, the shipped ensemble leg | 0.9170 | [0.9056, 0.9282] | Yes — **but contaminated, see below** |

For context, from the main pipeline's own `experiments/log.csv` (same T1, same metric code):

| Main-pipeline configuration | T1 MAP@3 | 95% CI |
|---|---|---|
| Own full-corpus BM25 open-book, best ckpt | 0.3869 | [0.3681, 0.4058] |
| cdeotte-context, source-matched, `deberta-v3-base`, best ckpt | 0.6086 | [0.5880, 0.6291] |
| — same run, **final** checkpoint | 0.3721 | [0.3534, 0.3913] |
| Closed-book `deberta-v3-base`, best ckpt | 0.5641 | [0.5439, 0.5840] |

The modernization is measured on a different axis (retrieval-level in-window evidence recall,
not MAP@3) and is reported in its own section below — short version: **+0.0153
[+0.0007, +0.0293]**, real but at the edge of resolvability, and the measurement redirects
the effort to context length instead.

### The faithful reproduction does not clear random, and that is the honest result

Reproducing the notebook **as written** gives **0.3770 [0.3577, 0.3960]** on T1. The random
baseline sits inside that CI, so this is **not resolved by this eval** — it is not a win
over guessing. It is also below the main pipeline's zero-parameter length heuristic (0.4780).

The training loss never went below `ln(5) = 1.6094` (per-window means: 1.681, 1.785, 1.780,
1.716, 1.687, 1.643, 1.643, 1.631), i.e. the model stayed at chance-level confidence
throughout. Loss *rose* over the first ~48 optimizer steps before cosine decay pulled it
back, which is the signature of too high a learning rate for the number of steps available.

Three things this is **not**, each checked rather than assumed:

- **Not a tokenization bug.** Decoded a real training example: exactly one `[CLS]`, two
  `[SEP]`, `token_type_ids` spanning both segments, sequence exactly 256 tokens, context
  truncated and option intact. cdeotte's literal-`[CLS]`-string trick does map to the real
  special-token ids under `transformers` 5.14.1.
- **Not a numerical failure.** Ran to completion with no non-finite loss (guard armed).
- **Not a leaked-eval artifact.** Asserted at both prep and train time.

**The most likely explanation is that the notebook's published configuration is a demo, not
the scoring configuration.** `NUM_TRAIN_SAMPLES = 1_024` sits under the comment
`# TRAIN WITH SUBSET OF 60K` — 1.7% of the available 60,347 rows — and part 1's own markdown
conditions the `CV MAP@3 = 0.915+` claim on "**by adjusting the parameters in this
notebook**". So the literal artifact and the quoted score are not the same configuration, and
a faithful reproduction of the literal artifact should not be expected to reach that number.
That gap is invisible unless you actually run it, which is the argument for running it.

Corroboration worth noting: **0.3770 is statistically indistinguishable from the main
pipeline's own final-checkpoint result on the same context, 0.3721 [0.3534, 0.3913]** —
different model size (large vs base), different lr (2e-5 vs 5e-6), different context budget
(256 vs 384), same answer. The main pipeline reached 0.6086 on this context only via
**best-checkpoint selection** against a sharp early transient; its end-of-training state
collapsed to 0.3721. cdeotte sets `load_best_model_at_end=False`, so a faithful
reproduction has no selection step and lands on the collapsed state. **On this data the
gap between 0.37 and 0.61 is checkpoint selection, not architecture or scale** — which
also means the main pipeline's 0.6086 depends on a selection step whose optimism is not
priced into its CI.

### The shipped checkpoint scores 0.9170 — and it is contaminated

`mgoksu/llm-science-run-context-2` is the second leg of part 2's 50/50 ensemble and one of
the two models the published 0.823761 actually ran. Scored on our T1 under part 2's own
inference format (context clipped to 1750 chars, `truncation=True`, no `max_length`;
measured sequence length mean 468, median 458, max 673 tokens): **0.9170 [0.9056, 0.9282]**.

**Do not read that as a calibration anchor.** Its training dataset,
`mgoksu/llm-science-exam-dataset-w-context`, contains the files `6000_train_examples.csv`
and `extra_train_set.csv` — the exact two radek1 files this project built T1 dev from.
Audited directly:

| mgoksu training file | rows | T1 dev prompts present |
|---|---|---|
| `6000_train_examples.csv` | 6,000 | 1,397 |
| `extra_train_set.csv` | 500 | 103 |
| **union** | | **1,500 / 1,500 = 100.0%** |

Every single T1 evaluation row is in that model's training set. 0.9170 measures
memorization. (Its dataset also ships a `train.csv`, so it very likely trained on the
official 200 as well.)

**This invalidates a strategy `PLAN.md` currently recommends.** `PLAN.md` proposes
reproducing a public notebook and measuring it on our holdout to get a `(holdout, LB)`
calibration pair — "your only bridge to it if late submission is disabled." That does not
work for this competition's strong public checkpoints: they were trained on the union of
the public synthetic pools, and any synthetic dev set built from those pools (T1 is, by
construction) is inside their training data. The only uncontaminated eval tiers for a 2023
public checkpoint are the **gold 200** (sacred here, capped) and **T3** (human-written
ARC/MMLU). Since late submission was already verified to score normally on 2026-07-29,
nothing is lost by dropping the anchor idea — but it should be dropped explicitly rather
than left in the plan as a fallback.

## What I could not reproduce, and why

**The retrieval half.** Part 2's two-stage dense retrieval needs
`jjinho/wikipedia-20230701` (~13 GB of parquet) plus the prebuilt article-level FAISS index
`jjinho/wikipedia-2023-07-faiss-index` (a single **9.66 GB** file) — both sizes verified with
`kaggle datasets files`. That does not fit this box's budget, and CLAUDE.md forbids building
a global dense index in local RAM (15 GiB total).

**The adaptation.** `cdeotte/60k-data-with-context-v2` already contains mgoksu-retrieved
context for all 60,347 rows, and **all 1,500 T1 dev prompts are present in it** (1,509 rows;
93.1% from source 2). So T1's context is cdeotte's *own retrieval output*, attached by
joining on prompt text. Train context and eval context therefore come from the same
retriever run by construction — CLAUDE.md's train/eval retriever-consistency rule holds
exactly, and it matches the original, where train and test context both came from running
the same notebook at `NUM_TITLES_INCLUDE=5, NUM_SENTENCES_INCLUDE=20`.

What that costs, stated plainly: **this reproduces the reader recipe and the
truncation/ordering behaviour, not the retrieval code.** No retrieval-quality claim here is
mine; it is inherited from mgoksu's run.

**The leakage trap that creates.** Because every T1 prompt is in the 60k, cdeotte's literal
`df_train.sample(1024)` would pull roughly `1024 x 1509/60347 ≈ 26` T1 rows into training.
`prepare_data.py` drops all 1,509 T1-matching rows before sampling and asserts the result.
Separately verified: **0 of the official gold 200 prompts appear in the 60k**, so the gold
set was never at risk.

## Incidental findings about the source data and environment

Recorded because they cost real debugging time and are not mentioned in the notebook.

- **`cdeotte/60k-data-with-context-v2` is noisier than its reputation.** 22.7% of rows have
  at least one null option (cdeotte's `.fillna('')` turns them into empty strings). The
  **correct** option is never empty (0.0%), so nulls are always distractors — meaning
  affected training rows are effectively easier than T1, whose rows all have five real
  distractors. There are also **6,138 duplicate-prompt rows** (60,347 rows, 54,209 distinct
  prompts), and **source 5 is a 100% subset of source 3** — all 5,920 of its distinct
  prompts reappear there, so the "12 concatenated datasets" double-count radek1's data.
- **Truncation, not retrieval, decides what the reader sees.** T1's context is a median
  **1,067 tokens** (mean 1,350) against `MAX_INPUT=256`, and **100% of rows exceed 256**. So
  `truncation='only_first'` discards roughly three quarters of every retrieved context, and
  because mgoksu concatenated the 20 stage-2 sentences in descending MiniLM order, which
  ~5–6 sentences survive is decided entirely by a 22M-param 2019 bi-encoder. This is the
  measurement that motivated the modernization below.
- **Part 2 ships a train/inference length mismatch.** Training uses 256 tokens; part 2's
  inference path clips context to 1750 characters and calls `truncation=True` with no
  `max_length`, which measured out at a mean of **468** tokens — 1.8x what training saw. The
  markdown attributes the clip to fixing a submission OOM, so a memory patch quietly became
  a distribution shift. (This project hit the mirror image on Day 2.)
- **`AdamW`'s default `eps=1e-8` produces NaN on this GPU.** With a healthy loss (1.54) and
  pre-clip grad norm (1.55), the *first* optimizer step turned all 103 trainable tensors
  non-finite — identically in fp32 and bf16, so not a precision issue. Isolated in
  `reproduced/debug_optim.py`: default and `foreach=False` both fail; **`fused=True` and
  `eps=1e-6` both survive**. `DEVLOG.md` independently root-caused the same failure as
  Adam's first-step instability and settled on `eps=1e-6` (Microsoft's own DeBERTa recipe),
  which is what this track uses — keeping the optimizer identical to the main pipeline's so
  the recipe stays the only difference between them. The `fused=True` datapoint is new.
- **Sequence-length note for anyone rerunning this:** frozen `deberta-v3-large` at
  batch 1 x 5 choices x 256 tokens measured 190 ms/micro-step and peaked at 5.40 GB of the
  8.55 GB card — it fits, but only because embeddings and 18 layers are frozen (77.2M of
  435.1M params trainable). Unfrozen, AdamW state alone is ~3.5 GB.

## Modernization: a 2026 cross-encoder reranker

**Built, applied to all frames, and measured at the retrieval level. The reader-level
MAP@3 delta was deliberately NOT run — the retrieval-level measurement says it is not
worth the compute, and that reasoning is given below with the numbers behind it.**

**The change.** Insert `Alibaba-NLP/gte-reranker-modernbert-base` (150M measured,
Apache-2.0, ModernBERT backbone so fp16-safe on Turing) between retrieval and the reader,
reordering the already-retrieved sentences by cross-encoder relevance to
`prompt + " " + A B C D E` (mgoksu's own stage-2 query). `PLAN.md`'s named reranker pick,
from its "2026 stack" section.

**Why this one.** The measurement above: with a median 1,067-token context against a
256-token budget, ~14 of 20 retrieved sentences never reach the reader, and MiniLM's
bi-encoder ranking is the sole arbiter of which do. A cross-encoder scores query and
sentence jointly rather than comparing independently-pooled vectors. It cannot add
evidence — it is strictly bounded by mgoksu's stage-1/stage-2 recall — it can only make
better use of the window. The alternative 2026-stack item, a stronger embedder, would
require re-running stage 1 over the 13 GB corpus, which is exactly what is not available.

**Consistency requirement.** Reranking only the eval context while training on
MiniLM-ordered context would manufacture the train/inference mismatch this project spent
Day 2 diagnosing. So both the training frames and the eval frame are reranked and the
reader is **retrained**, giving a clean paired comparison on identical T1 rows.

**Baseline choice.** The delta is measured against a faithful-recipe run at the **same
8,192-row scale**, not against the 1,024-row literal reproduction. Measuring a reranker
delta against a baseline that sits inside the random-noise band would be uninterpretable —
a reader that has not learned to use its context cannot demonstrate sensitivity to context
ordering. Scaling the training set is the smallest change that gives the comparison a
chance of being meaningful, and it is a knob the notebook itself flags as adjustable.

**Honest imperfection.** The stored `context` is 20 blingfire sentences joined with spaces;
`blingfire` is not installed here and adding it would mean modifying the shared venv, so
the units are recovered with a regex splitter. It over-splits — mean **27.4** units per row
against the original 20 (e.g. isolating a bare "1940.") — so the reranker reorders slightly
finer-grained units than mgoksu retrieved, and this is a reordering of *approximately*, not
exactly, the original sentences. Only 2 of 1,500 rows yielded ≤1 unit (rerank a no-op).

**Cost.** 150M params, 1.79 GB peak, measured 4.9–5.1 rows/s end-to-end (~27 pairs per row).
Batching pairs across rows rather than one row at a time was needed to make it tractable;
even so it is GPU-bound, not overhead-bound. Reranking 10,192 rows costs ~35 min on this
card — an offline, one-time cost that adds nothing to reader latency.

### Result: the mechanism works, but the headroom is tiny

`reproduced/window_recall.py` tests the modernization's causal mechanism **without a reader
and without training** — far cheaper (~2 min, CPU) and cleaner than inferring the mechanism
from a MAP@3 delta. For each T1 row it takes the context that actually survives
`truncation='only_first'` at the reader's real budget (256 tokens minus that row's
`#### prompt [SEP] option [SEP]` segment), under MiniLM order vs reranker order, and asks
whether the answer-supporting evidence is inside it — reusing `llmsci.retrieve.eval`'s own
`distinctive_keywords` / `is_answer_support_hit`, so the number is directly comparable to
the main pipeline's Proxy B retrieval numbers.

Answer-support hit rate inside the 256-token window, n=1500, paired:

| Context ordering | in-window hit rate | 95% CI |
|---|---|---|
| MiniLM (as shipped) | 0.5613 | [0.5360, 0.5867] |
| `gte-reranker-modernbert-base` | 0.5767 | [0.5513, 0.6020] |
| **paired delta** | **+0.0153** | **[+0.0007, +0.0293]** |
| Full untruncated context | 0.6527 | [0.6287, 0.6773] |

Mean context surviving truncation: **218 tokens** of a median 1,067 — i.e. ~80% discarded,
which confirms the mechanism claim quantitatively rather than by assertion.

Three things follow, and the third is the decision:

1. **Truncation is expensive**: it costs **9.1 points** of answer-support recall
   (0.6527 → 0.5613). The evidence is retrieved and then thrown away by the 256-token budget.
   This is the single largest identified loss in cdeotte's pipeline.
2. **The reranker does help, but recovers only 1.5 of those 9.1 points** — about 17% of the
   truncation gap, changing hit status on just **8.1%** of rows. Its CI excludes 0, so it is
   *technically* resolved, but the lower bound is **+0.0007** and the delta (+0.0153) barely
   exceeds this eval's minimum detectable effect (**±0.0144**). Per CLAUDE.md this should be
   read as a real-but-marginal effect at the edge of what 1,500 rows can resolve, **not** as
   a clean win.
3. **Therefore the reader-level experiment was not run.** A +0.015 gain in in-window
   evidence recall cannot plausibly produce a MAP@3 gain that a paired bootstrap on 1,500
   rows could resolve — MAP@3 deltas convert at less than 1:1 from recall, and `PLAN.md`'s
   own noise arithmetic puts the resolvable floor for a small change at roughly ±0.018–0.02.
   Spending two 33-minute training runs to report an unresolved null would have burned an
   hour of GPU to learn less than the 2-minute measurement already established.

**The more interesting implication is about which 2026 upgrade to pick.** The measured
bottleneck is not ranking, it is the **256-token budget**: reordering within the window buys
1.5 points while the window itself costs 9.1. `PLAN.md` already identifies context length as
"the highest-ROI knob" and explains why DeBERTa-v3 can run at 1280 tokens with no model
surgery (`position_biased_input: false`, relative attention only). This measurement supports
that ordering on cdeotte's pipeline specifically: **raise `MAX_INPUT` before adding a
reranker.** A reranker earns its keep when the candidate pool is large and the window is not
the binding constraint; here the window is the binding constraint.

One caveat on the absolute numbers: 346 of 1,500 rows have no distinctive keyword, so the
metric is undefined for them and they are counted as misses. That depresses both arms
equally and cancels in the paired delta, but it means 0.5613 / 0.6527 understate true
support recall.

## Assessment

**On the reproduction.** The literal published notebook, faithfully reimplemented, scores
**0.3770 [0.3577, 0.3960]** on our T1 — not resolved against the 0.3667 random baseline.
That is a real finding about the artifact, not a failure of the reproduction: the notebook's
`NUM_TRAIN_SAMPLES=1_024` is a demo subset and its own markdown conditions the 0.915 claim
on "adjusting the parameters". The published **0.823761 is also a 2-model ensemble**, and
part 1 claims a *higher* single-model LB (0.830+), so the ensemble did not help on the
public leaderboard either.

**On what the comparison actually bought.** The most valuable output of this track is not a
score, it is the **contamination audit**: 100% of T1 dev is inside the shipped checkpoint's
training data, which retires `PLAN.md`'s "reproduce a notebook for a (holdout, LB)
calibration anchor" idea for this competition. The second most valuable is the finding that
the 0.37-vs-0.61 gap on identical context is **checkpoint selection**, which reframes the
main pipeline's own 0.6086 as selection-dependent in a way its CI does not capture.

**On the modernization.** The reranker's mechanism is real but its headroom is small:
+0.0153 [+0.0007, +0.0293] in-window evidence recall, at the very edge of what 1,500 rows
resolve. **Honest answer: it did not help enough to be worth shipping here**, and the
measurement that establishes that also redirects the effort — truncation costs 9.1 points
where reranking recovers 1.5, so the 256-token budget is the thing to fix first. Reporting
the cheap retrieval-level measurement and *declining* the expensive reader-level one is the
right trade, and it is the opposite of the failure mode where a reranker gets added because
it is fashionable and its gain is never isolated.

**On whether the main pipeline should reference this.** Reference the contamination finding
and the checkpoint-selection finding. Do **not** cite 0.9170 as any kind of ceiling or
anchor. The reproduction's own 0.3770 is worth one row in a comparison table, clearly
labelled as the literal-notebook configuration rather than as "cdeotte's 0.82 pipeline".
The truncation-vs-reranking result is worth a line in the retrieval report, since it is a
measurement on a *different* pipeline than the main one and so is corroborating evidence for
`PLAN.md`'s context-length claim rather than a result about our own retriever.

## Reproducing this

```
cd reference_reproduction/reproduced
export PYTHONPATH=<repo>/src            # for llmsci.metrics and llmsci.gpu_guard
python prepare_data.py --n-train 8192   # leakage-guarded frames + T1 context join
python bench_step.py                    # measures the ms/step the WSL2 guard uses
python rerank.py                        # modernization: adds context_reranked (~35 min)
python window_recall.py                 # RUN THIS FIRST: cheap mechanism test, ~2 min
python train.py --tag faithful_8192  --context-col context           --train-file train_8192.parquet --epochs 1
python train.py --tag reranked_8192  --context-col context_reranked  --train-file train_8192.parquet --epochs 1
python compare.py faithful_8192 reranked_8192
python eval_shipped.py --model-dir ../models/mgoksu-run-context-2 --tag mgoksu_shipped
```

Order matters: `prepare_data.py` rewrites the T1 parquet without `context_reranked`, so
`rerank.py` must run after it, and `window_recall.py` / the `reranked` training run after that.

The literal-notebook reproduction behind row 2 of the headline table was
`prepare_data.py` (default `--n-train 1024`) then
`train.py --tag faithful --context-col context`, i.e. cdeotte's 2 epochs at his own
`NUM_TRAIN_SAMPLES`.

`data/`, `models/`, and `results/` here hold gitignored artifacts (parquet, checkpoints).
Logs from the runs behind this report are the `results_*.log` files in this directory.

## Status

- [x] Notebook pulled and read end to end (`original/`, `NOTES.md`)
- [x] Reader reimplemented from the notes; literal-recipe T1 score with CI (**0.3770**)
- [x] Shipped `mgoksu` checkpoint scored on T1 (**0.9170**) and contamination-audited
      (**100% of T1 inside its training data**)
- [x] Incidental data-quality and environment findings, all measured
- [x] Reranker implemented and applied to **all three frames** (T1, train 8,192, monitor 500;
      29 min total, peak 1.79 GB)
- [x] Modernization measured at the retrieval level: in-window answer-support recall,
      paired, with CI and MDE (**+0.0153 [+0.0007, +0.0293]**)
- [ ] Reader-level MAP@3 for the reranked variant — **deliberately not run**, on the basis of
      the measurement above. This is a reasoned decline, not an unfinished step: the two
      33-minute runs would have been spent to report a delta this eval cannot resolve.

Item 4 is answered at the mechanism level and honestly declined at the reader level. If
someone wants the reader-level number anyway, both commands are in "Reproducing this" and
all inputs are on disk.

The one genuinely open thread, worth more than the reranker retrain: **raise `MAX_INPUT`
from 256 and re-measure**, since truncation is where the 9.1 points went.
