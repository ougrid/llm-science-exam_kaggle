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

- `microsoft/deberta-v3-base` and `microsoft/deberta-v3-large` (MIT) — the
  reader models for every training run in this project.
- `bm25s` (MIT) — sparse retrieval, `method="lucene"`.
- `transformers`, `datasets`, `accelerate`, `torch` (HuggingFace / PyTorch,
  Apache-2.0 / BSD) — training and modeling stack.

## Public checkpoints used in the final submission — NOT trained by me

Using public models and datasets was explicitly permitted by this competition
and was standard practice in it, so these are legitimate submission
components. They are **not** evidence of models I trained, and every number
derived from them is reported in its own clearly-labelled table row (see
`README.md`'s "Not mine" table and `reports/ablation_table.md`'s reference
table).

- **[`mgoksu/llm-science-run-context-2`](https://www.kaggle.com/datasets/mgoksu/llm-science-run-context-2)**
  — a `deberta-v3-large` `AutoModelForMultipleChoice` checkpoint fine-tuned by
  another competitor; one leg of the 50/50 ensemble behind
  `cdeotte/how-to-train-open-book-model-part-2`'s published 0.823761 public LB.
  **What I took:** the weights, used verbatim for inference, plus part 2's
  inference format (`context[:1750] + " #### " + prompt`, then
  `tokenizer(first, option, truncation=True)`), transcribed so the checkpoint is
  scored the way it was actually run.
  **What is mine:** the corpus, chunking, title-prefixing, BM25 index, query
  construction, the offline inference pipeline, and the evaluation protocol —
  including the finding that this checkpoint scores **0.9170 on my T1 dev set
  but is 100% contaminated against it** (audited at file level: its training
  files cover all 1,500 T1 prompts), which is why it is only ever reported on
  the clean gold 200 (0.8592 [0.8200, 0.8958]).
- **[`sandiago21/llm-science-exam-deberta-v3-large-context-3`](https://www.kaggle.com/datasets/sandiago21/llm-science-exam-deberta-v3-large-context-3)**
  — a second, independently-trained public checkpoint, attached as a candidate
  ensemble leg. **What I took:** the weights. **What is mine:** the
  ensembling and the selection protocol — all candidate configs scored on the
  clean gold 200 with bootstrap CIs, shipping only the config that measurably
  wins.

## Reproduction of a public solution (`reference_reproduction/`)

`cdeotte/how-to-train-open-book-model-part-1` / `-part-2` were reproduced in a
separate, clearly-labelled track following `PLAN.md`'s legitimate-reuse process
(read end-to-end, write down what it does in prose, reimplement from the prose,
diff the numbers, cite). See `reference_reproduction/NOTES.md` and
`RESULTS.md`.

**What that track took:** the two-stage retrieval idea, the frozen-layer recipe
(embeddings + first 18 of 24 layers, 77.2M of 435.1M trainable), and part 2's
inference tokenization.
**What it produced independently:** the literal published configuration scores
0.3770 [0.3577, 0.3960] on my dev set — i.e. it does not clear baseline either,
and its 0.823761 requires "adjusting the parameters" per its own markdown; the
null-option data bug; an AdamW `eps=1e-8` NaN failure and its `fused=True`
workaround; and the contamination audit above.

## What has NOT happened

No public solution's code was copied into `src/llmsci/` or into any submission
notebook. The 1st-place H2O LLM Studio solution and `mbanaei`'s TF-IDF pipeline
were **read only** (via their writeups) and are cited in `PLAN.md` as
calibration context and as the source of specific ideas — layer freezing,
multiple-corpus blending, per-option retrieval, training on retrieved rather
than oracle context. Ideas are attributed at the point of use in the code and
in `DEVLOG.md`.
