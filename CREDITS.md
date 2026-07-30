# Credits

External data, models, and tools this project builds on, and specifically
what was taken from each versus built independently. See `PLAN.md`'s
anti-patterns section for why this file exists: undisclosed reuse is a
liability, cited reuse is a credibility signal.

## Data

- **[`radek1/additional-train-data-for-llm-science-exam`](https://www.kaggle.com/datasets/radek1/additional-train-data-for-llm-science-exam)
  and `radek1/15k-high-quality-examples`** — the ~6.5k GPT-3.5-generated
  synthetic question pool this project's T1 dev set and closed-book/
  own-retrieval training data are built from (`scripts/build_eval_tiers.py`).
  Used as-is; the three-tier split (T1/T2/T3), leakage checks, and
  near-duplicate-vs-gold filtering are this project's own work.
- **[`cdeotte/60k-data-with-context-v2`](https://www.kaggle.com/datasets/cdeotte/60k-data-with-context-v2)**
  — used as a shortcut to get pre-retrieved training context (PLAN.md's
  explicit Day-2 suggestion, to skip building a training-scale retrieval
  pipeline before one exists). Found and fixed two real problems in it
  independently: 25.8% of rows have a null option, and its 12 constituent
  sources have a train/eval context-distribution mismatch when naively
  split — both documented in `DEVLOG.md` and `experiments/log.csv`, not
  silently worked around.
- **[`mbanaei/all-paraphs-parsed-expanded`](https://www.kaggle.com/datasets/mbanaei/all-paraphs-parsed-expanded)**
  — the ~276k-article, paragraph-parsed Wikipedia STEM corpus behind this
  project's own retrieval pipeline. The chunking (`src/llmsci/corpus.py`),
  title-prefixing, and BM25 indexing (`src/llmsci/retrieve/sparse.py`) are
  this project's own implementation, not copied from any reference
  notebook.
- **ARC-Challenge and MMLU-STEM** (via HuggingFace `datasets`) — used
  unmodified as the human-written T3 out-of-distribution eval tier, per
  PLAN.md's validation design.

## Ideas taken from prior public work (not code)

- **Title-prefixing each chunk before indexing/embedding** — the 1st-place
  H2O LLM Studio writeup's stated trick; reimplemented independently in
  `src/llmsci/corpus.py`, not copied from their code (which was read only
  at a high level for the writeup's stated approach, not line-by-line).
- **Hierarchical article→chunk retrieval, and treating the 200 official
  rows as a sacred held-out set** — general ideas from `PLAN.md`'s own
  research pass, not attributable to a single source.

## Models and tools

- `microsoft/deberta-v3-base` (MIT) — the reader model for every training
  run in this project so far.
- `bm25s` (MIT) — sparse retrieval, `method="lucene"`.
- `transformers`, `datasets`, `accelerate`, `torch` (HuggingFace / PyTorch,
  Apache-2.0 / BSD) — training and modeling stack.

## What has NOT happened (stated explicitly, per PLAN.md's own anti-pattern
warning against silent omission)

No public solution notebook for this competition (1st place, `cdeotte`'s
open-book notebooks, `mbanaei`'s TF-IDF pipeline, etc.) has been read,
reimplemented, or reused as code. Their existence and headline scores are
cited in `PLAN.md` as calibration context, not as a source this project's
own pipeline derives from. If that changes later (e.g. reproducing one for
a calibration anchor per `PLAN.md`'s Day-2 plan), it will be logged here
following `PLAN.md`'s stated process: read end-to-end, close the tab,
reimplement from a written description, diff the numbers, and only then
cite it.
