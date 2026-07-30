# What cdeotte's open-book notebooks actually do

Reading notes for the comparison track. Written after reading both notebooks end to end
and before writing any code in `reproduced/`, per `PLAN.md`'s legitimate-reuse process
(read it → write down what it does in prose → reimplement from the prose).

**This is not part of the main pipeline.** Nothing here feeds `src/llmsci/`,
`experiments/log.csv`, or the headline ablation table. It exists to answer one question:
what does a known-good 2023 public solution score *on our own T1 dev set*, so that our
own pipeline's numbers have an external reference point measured on the same ruler.

Sources, both pulled with `kaggle kernels pull` on 2026-07-30 (raw `.ipynb` in
`original/`, unmodified):

- `cdeotte/how-to-train-open-book-model-part-2` — public LB **0.823761**.
  https://www.kaggle.com/code/cdeotte/how-to-train-open-book-model-part-2
- `cdeotte/how-to-train-open-book-model-part-1` — the training half that part 2 consumes.
  Claims `CV MAP@3 = 0.915+` and "single model LB 0.830+".
  https://www.kaggle.com/code/cdeotte/how-to-train-open-book-model

The two are one pipeline split across the internet boundary: part 1 trains with internet
on, part 2 runs retrieval + inference with internet off so it can be submitted. Part 2 is
itself a fork of `mgoksu/0-807-sharing-my-trained-with-context-model`, whose retrieval
code descends from `jjinho/open-book-llm-science-exam`. The lineage matters because the
retrieval half is essentially unchanged from jjinho/mgoksu — cdeotte's contribution is the
reader training recipe and the ensemble.

## Retrieval (part 2, inherited from mgoksu/jjinho)

Two-stage dense, article then sentence. No sparse leg anywhere, no fusion, no reranker.

**Corpus.** `jjinho/wikipedia-20230701` — a ~270k-article STEM-ish Wikipedia plaintext
parse, sharded into parquet files by first letter of title (`a.parquet`, `b.parquet`, …,
plus `number.parquet` and `other.parquet`), with a `wiki_2023_index.parquet` mapping
article `id` → which shard file holds it.

**Embedder for both stages.** `sentence-transformers/all-MiniLM-L6-v2`, 384-d, run in
fp16 with `max_seq_length=384`, `normalize_embeddings=True`. One model does article
retrieval and sentence retrieval — no separate query/document encoders and no instruction
prefixes (MiniLM has none).

**Stage 1 — article retrieval.** A prebuilt FAISS index of article-level MiniLM
embeddings (`jjinho/wikipedia-2023-07-faiss-index`, a single 9.7 GB `.index` file) is
searched with the **`prompt` alone** — the five options are *not* used here. Top 5
articles per question (`NUM_TITLES_INCLUDE = 5`).

**Stage 2 — sentence retrieval within those articles.** Full text of the ~5 retrieved
articles per question is loaded from the parquet shards, split into sentences with
`blingfire` (`text_to_sentences_and_offsets`, dropping sentences ≤ 3 chars), and every
sentence is MiniLM-encoded. Then, **per question**, a fresh flat FAISS index is built over
only that question's candidate sentences and searched with `prompt + " " + A B C D E`
(all five options concatenated — so the options *are* used at stage 2, just not stage 1).
Top 20 sentences (`NUM_SENTENCES_INCLUDE = 20`) are concatenated in descending
similarity order, space-separated, into a single `context` string.

Two things worth flagging as design weaknesses, since they're where a 2026 stack would
attack:

1. **Stage 1 queries with the prompt only.** The options carry the rare anchor terms —
   the entity names, the units, the reaction products. Dropping them at the article stage
   means an article-level miss can never be recovered downstream, and stage 2 can only
   rerank within whatever 5 articles stage 1 happened to return. `PLAN.md` makes the same
   point in "Query construction matters as much as the index."
2. **Sentence order is the only thing protecting the evidence.** See the truncation
   interaction below — this turns out to be the dominant mechanism.

Also worth noting: the per-question FAISS index in cell 23 is searched with the *full*
`question_embeddings` matrix and then indexed at `[prompt_id]`, i.e. it computes all
N queries against each question's sentence pool and throws away all but one row. That's
O(N²) wasted work, harmless at N=200 but it is why the notebook is slow. And `context` is
carried over from the previous loop iteration when a question retrieves zero sentences
(the `context = ""` initialisation sits *inside* the `if`), a latent bug that doesn't fire
on the competition test set.

## Reader (part 1)

`AutoModelForMultipleChoice` on `microsoft/deberta-v3-large` (435M). Standard shared-scalar
MC head — `(bs, 5, L)` in, `(bs, 5)` logits out, cross-entropy against the correct index.
Same reader family as our own pipeline, which is why this was chosen as the comparison
target over the TF-IDF `mbanaei` notebook.

**Input construction** (the part that is easy to get subtly wrong):

```
first_sentence  = "[CLS] " + context                       (repeated 5x)
second_sentence = " #### " + prompt + " [SEP] " + option + " [SEP]"   (one per option)
tokenizer(first, second, truncation='only_first', max_length=256, add_special_tokens=False)
```

The special tokens are written as *literal text* and `add_special_tokens=False`, which
reproduces what `add_special_tokens=True` would emit anyway (the tokenizer maps the
`[CLS]`/`[SEP]` strings to their real ids). The reason for the manual version is control:
it lets `truncation='only_first'` cut the **context** and never the question or the option.
`PLAN.md` lists `truncation=True` instead of `'only_first'` as a silent-capping gotcha, and
this notebook gets it right.

**The interaction that dominates everything.** `MAX_INPUT = 256`, but the retrieved
context is far longer than that — measured on our T1-matched rows, the median context is
**1067 tokens** and *100%* of rows exceed 256. So `truncation='only_first'` throws away
roughly three quarters of the retrieved context, and because stage-2 concatenated the 20
sentences in descending MiniLM similarity order, **which ~5–6 of the 20 retrieved
sentences the reader actually sees is decided entirely by MiniLM's bi-encoder ranking.**
That is the highest-leverage point in the whole pipeline and it is the one I target for
the modernization.

**Training recipe.**

| | |
|---|---|
| Train data | `cdeotte/60k-data-with-context-v2` → `all_12_with_context2.csv`, 60,347 rows, 12 concatenated public synthetic datasets, `context` already retrieved by running mgoksu's notebook over the whole 60k |
| Subsample | `NUM_TRAIN_SAMPLES = 1_024`, plain random `.sample()` — only 1.7% of the 60k |
| Missing options | `.fillna('')` |
| Validation | `train_with_context2.csv` = the official 200 `train.csv` rows with context added |
| Frozen | embeddings, plus the first **18 of 24** encoder layers → ~6 layers trainable |
| Optimizer | lr 2e-5, `warmup_ratio=0.1`, cosine schedule, `weight_decay=0.01` |
| Batch | `per_device_train_batch_size=1`, `gradient_accumulation_steps=8` → effective 8 (×2 GPUs) |
| Epochs | 2, fp16 |
| Checkpointing | eval + save every 25 steps, `save_total_limit=2`, **`load_best_model_at_end=False`** |

Two observations on the recipe. First, the freezing is not a quality choice, it's a
memory choice — cdeotte's own markdown lists freezing under "tricks to train models
efficiently" on Kaggle and says validation accuracy "may become less". Second,
`load_best_model_at_end=False` means the *final* checkpoint is what gets saved and
shipped, so the quoted `CV MAP@3 = 0.915+` is a number read off the eval-every-25-steps
log, not necessarily the score of the model that was saved. That's a soft version of the
best-of-N optimism `PLAN.md` warns about — and the validation set it's read off is the
official 200, which is precisely the set our own project treats as sacred and
never-tuned-against. Worth stating plainly: **cdeotte's 0.915 CV and our T2 gold number
are not the same kind of number**, and that alone justifies re-measuring on our own T1.

## Inference and ensemble (part 2)

Context is clipped to the first **1750 characters**, then `context[:1750] + " #### " + prompt`
becomes the `prompt` field, and inference tokenizes with
`tokenizer(first_sentence, second_sentence, truncation=True)` — plain `truncation=True`,
**no `max_length`**, and `add_special_tokens` left at default. So the token layout matches
training (`[CLS] ctx #### prompt [SEP] option [SEP]`) but the *length budget does not*:
training saw 256 tokens, inference lets the tokenizer run to its own `model_max_length`.
The 1750-character clip is the real limiter and it corresponds to ~400 tokens, i.e.
inference feeds the reader noticeably more context than training ever showed it. The
markdown says the clip was added to fix an OOM/"CSV not found" submission failure, so it's
a memory patch that quietly became a train/inference length mismatch. Our own project hit
the mirror image of this on Day 2 (the "mismatch row"), so it is not a criticism so much
as confirmation that this is the easiest bug in the genre to ship.

Batch size 1 at inference, no length bucketing.

**Ensemble.** Logits from two DeBERTa-v3-large models are averaged 50/50: cdeotte's own
part-1 model, and `mgoksu/llm-science-run-context-2` (the pretrained checkpoint from the
notebook part 2 forked). Top 3 letters by descending averaged logit. So the published
**0.823761 is a 2-model ensemble score**; part 1's claimed single-model LB is 0.830+,
which is *higher* — the ensemble did not help on the public LB.

## What I can and cannot reproduce here, and how I adapted

Constraints: one 8 GB RTX 5050 under WSL2, 15 GiB system RAM, a few hours.

**Not reproducible as written — the retrieval half.** `jjinho/wikipedia-20230701` is
~13 GB of parquet and the prebuilt article FAISS index is a single **9.7 GB** file. Both
numbers verified with `kaggle datasets files`. Downloading and searching that locally is
out of budget, and `CLAUDE.md` forbids building a global dense index in local RAM anyway.

**The adaptation that rescues it.** `all_12_with_context2.csv` already contains
mgoksu-retrieved context for all 60,347 rows, and our T1 dev set was built from radek1's
pools — which are sources 1–6 of cdeotte's 12. Measured: **all 1,500 T1 dev prompts are
present in `all_12_with_context2.csv`** (1,509 rows; 9 prompts appear twice), 93% of them
from source 2. So I can attach *cdeotte's own retrieval output* to every T1 dev row by
joining on prompt text, instead of re-running a retriever I can't afford to run.

This is better than a re-implementation would have been, on the axis this project cares
about most: train context and eval context then come from **the same retriever run** by
construction, which is exactly `CLAUDE.md`'s "retrieval training contexts must come from
the same retriever used at test time" rule. It is also faithful to the original, where
train and test context both came from running the same notebook code with the same
`NUM_TITLES_INCLUDE=5, NUM_SENTENCES_INCLUDE=20`.

What it costs: I am reproducing the **reader recipe and the truncation/ordering
behaviour**, not the retrieval code. Any retrieval-quality claim about this pipeline is
inherited from mgoksu's run, not measured by me. Stated as a limitation, not papered over.

**The leakage trap this creates, and the guard.** Because every T1 prompt is in the 60k,
cdeotte's literal `df_train.sample(1024)` would pull roughly `1024 × 1509/60347 ≈ 26` T1
dev rows straight into training. That is small but it is exactly the contamination
`PLAN.md`'s leakage section is about, and it would inflate the very number I'm trying to
report. My `prepare_data.py` therefore drops all 1,509 T1-matching rows from the training
pool before sampling. Separately verified: **0 of the official gold 200 prompts appear in
`all_12_with_context2.csv`**, so the gold set is not at risk and I do not need to touch it
(and won't — no gold evaluation is spent on this comparison track).

**Other adaptations, all forced by environment rather than chosen:**

- **`transformers` is 5.14.1 here, the notebook was written against 4.31.** `PLAN.md` flags
  this. `evaluation_strategy` → `eval_strategy`, `Trainer(tokenizer=)` →
  `Trainer(processing_class=)`.
- **bf16, not fp16.** Blackwell has bf16; and DeBERTa-v3's `layer_norm_eps=1e-7` is a known
  fp16 NaN source. The notebook's `fp16=True` was for T4s, which have no bf16.
- **Single GPU, not 2×T4.** cdeotte's effective batch is 8 × 2 GPUs = 16; I use
  `gradient_accumulation_steps=16` on one GPU to match the effective batch rather than
  matching the literal accumulation number.
- **`gpu_guard` before anything loads.** `cap_memory_fraction()` +
  `assert_step_speed()`/`probe_training_speed()`, per `CLAUDE.md` — WSL2 will silently
  spill past 8.55 GB into system RAM at 15–25× slowdown with no error.
- **Validation set swapped.** cdeotte evaluates during training on the official 200. Our
  200 are sacred, so I hold out a slice of the training pool for the learning curve. Since
  `load_best_model_at_end=False`, nothing is selected on it, so this changes no result —
  it's monitoring only. T1 is scored **once**, on the final checkpoint.
- **Ensemble reproduced only if cheap.** The second leg is a 2023 Kaggle-hosted DeBERTa-v3-large
  checkpoint; whether it still downloads is a live question. The single-model number is
  the primary result either way, and part 1 claims single-model ≥ ensemble on LB anyway.

## The modernization I'm going to try

One change, from `PLAN.md`'s "2026 stack" — **insert a cross-encoder reranker between
retrieval and the reader**, `Alibaba-NLP/gte-reranker-modernbert-base` (149M, Apache-2.0,
ModernBERT backbone so fp16-safe, `PLAN.md`'s named reranker pick).

Why this one and not a stronger embedder: swapping the embedder means re-running stage-1
and stage-2 retrieval, which needs the 13 GB corpus I don't have. Reranking operates on
the **20 sentences already retrieved**, which are recoverable from the stored `context`
string, so it's the only 2026-stack item that fits the data I actually have.

Why it should matter, mechanistically — the measurement above: context is median 1067
tokens against a 256-token budget, so ~14 of the 20 retrieved sentences are discarded by
truncation, and MiniLM's bi-encoder ranking is the sole arbiter of which survive. A
cross-encoder scores `question + options` against each sentence *jointly* rather than
comparing two independently-pooled vectors, which is where reranking earns its keep. This
raises precision@5 without touching recall@20 — exactly the framing in `PLAN.md`'s
Day 3 item 3. Note the reranker's job here is unusually pure: it cannot add evidence, only
reorder what MiniLM already found, so it is strictly bounded by mgoksu's stage-1/stage-2
recall.

Implementation: split the stored `context` back into sentences, score each against
`prompt + " " + A..E` with the reranker, re-concatenate in descending reranker score.
The re-split is approximate — the originals were blingfire-split then joined with a space,
so a regex splitter won't recover them exactly. Honest imperfection, recorded.

**The consistency requirement this drags in:** if I rerank the eval context but train on
MiniLM-ordered context, I have manufactured precisely the train/inference mismatch this
project spent Day 2 on. So the modernization requires **retraining** on reranked context
and re-scoring — two full runs, not one, compared with a paired bootstrap on the same
1,500 T1 rows.

> **Outcome (added after execution — this file is the pre-implementation plan, kept as the
> record of what was intended).** The two retraining runs were *not* spent. A cheaper
> retrieval-level measurement (`reproduced/window_recall.py`) showed the reranker recovers
> only +0.0153 [+0.0007, +0.0293] of in-window evidence recall while truncation itself costs
> 9.1 points — too small for a reader-level MAP@3 delta to be resolvable on 1,500 rows. See
> `RESULTS.md`, "Modernization", for the numbers and the reasoning behind declining.
