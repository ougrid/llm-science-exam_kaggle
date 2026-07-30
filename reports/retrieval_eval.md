# Retrieval eval harness results (Day 3, in progress)

`src/llmsci/retrieve/eval.py`, run via `scripts/run_retrieval_eval.py` against
the full-corpus BM25 index (`data/bm25_index_full`, 2,345,229 chunks over
~276k articles). All numbers are **Proxy B: answer-support recall** — a
retrieved chunk counts as a hit if it contains a content word (≥5 letters)
present in the correct option but absent from all four distractors. This
is a proxy, not ground truth (see caveats below), and it's the only proxy
usable here: T1, T3, and the gold 200 all lack a "generating article" label
(checked directly against `radek1`'s pool, `extra_train_set.csv`, and
`t3_ood.parquet`), so PLAN.md's source-article-based "Proxy A" isn't
computable with the data this project has.

## T1 (1,500 rows, synthetic, GPT-3.5-generated)

| metric | mean | 95% CI |
|---|---|---|
| recall@1 | 0.2760 | [0.2540, 0.2980] |
| recall@5 | 0.4813 | [0.4560, 0.5067] |
| recall@10 | 0.5653 | [0.5407, 0.5907] |
| recall@20 | 0.6193 | [0.5940, 0.6440] |
| recall@50 | 0.6840 | [0.6600, 0.7073] |
| recall@100 | 0.7107 | [0.6880, 0.7333] |
| MRR | 0.3718 | [0.3509, 0.3922] |
| nDCG@10 | 0.4133 | [0.3922, 0.4342] |

## T3-OOD (1,000 rows, human-written, ARC-Challenge + MMLU-STEM)

| metric | mean | 95% CI |
|---|---|---|
| recall@1 | 0.3040 | [0.2760, 0.3320] |
| recall@5 | 0.4500 | [0.4190, 0.4810] |
| recall@10 | 0.5090 | [0.4770, 0.5400] |
| recall@20 | 0.5430 | [0.5120, 0.5730] |
| recall@50 | 0.5740 | [0.5430, 0.6040] |
| recall@100 | 0.5880 | [0.5580, 0.6180] |
| MRR | 0.3720 | [0.3449, 0.3990] |
| nDCG@10 | 0.4020 | [0.3744, 0.4292] |

## Reading these numbers honestly

- **Recall plateaus well short of 1.0 on both tiers.** ~29% of T1 questions
  and ~41% of T3 questions never get an answer-supporting chunk even at
  k=100. Some of that is real retrieval failure (BM25 missing the right
  passage); some is proxy limitation (a genuinely relevant chunk can
  explain the answer without using the exact distinctive keywords this
  proxy looks for). Both directions are real — this number brackets rather
  than measures retrieval quality precisely.
- **T1 has meaningfully higher recall@50/100 than T3** (0.684/0.711 vs.
  0.574/0.588) despite similar recall@1 and nearly identical MRR. This is
  a real, measured distribution shift: GPT-3.5-generated questions (T1)
  are presumably lexically closer to the Wikipedia corpus they were
  generated from than human-written ARC/MMLU questions (T3) are — exactly
  the kind of shift `PLAN.md` built T3 to catch, now demonstrated rather
  than assumed.
- **This connects directly to the open-book training results — but not
  the way expected.** Training against this full-corpus index landed at
  0.3869 MAP@3, statistically indistinguishable from the 20k-article
  pilot slice's 0.3866, despite this index's measured recall@5=0.481
  being real, substantial signal. Hand-inspecting the actual retrieved
  text (not another aggregate metric) found why: for a *Didymogenes*
  classification question, cdeotte's context (used in the 0.6086
  source-matched run) opens with "Didymogenes is a genus of green algae
  in the class Trebouxiophyceae" — nearly verbatim the answer — while our
  own retrieval, even over the full corpus, returned a different genus
  (*Ochromonas*) entirely. For a "Big Mama Thornton" question, ours
  retrieved a Chinese internet-censorship neologism and an unrelated
  athlete, fooled by matching "Big" + "mama" + "Thornton" as three
  independent tokens rather than one named entity. **BM25's bag-of-words
  scoring has no phrase- or entity-level understanding, and that failure
  mode doesn't improve with more corpus coverage** — recall@k answers "is
  a supporting chunk retrievable at all," not "does it outrank
  lexically-similar-but-wrong chunks," which is the thing that actually
  determined training performance here. Full writeup in `DEVLOG.md`.

## Not yet done

- **Oracle-context ceiling** (feed the reader the *known-good* passage and
  measure MAP@3) needs a trained reader and a way to identify the "known
  good" passage per question — with no source-article ground truth, this
  needs its own design (e.g., use a high-confidence answer-support hit as
  a stand-in oracle passage) rather than PLAN.md's original recipe.
- **2×2 failure decomposition** (reader-correct × gold-passage-retrieved)
  needs the trained reader's predictions. The full-corpus retraining run
  has now completed (0.3869 MAP@3) — this is unblocked but not yet built.
- **Phrase-aware or entity-aware retrieval (e.g. a reranker), identified
  above as the actual next lever** — not corpus coverage, which is now
  ruled out by the 20k-vs-276k comparison. Requiring multi-word phrase
  matches to score highly, or reranking top-k with a cross-encoder that
  can recognize "Big Mama Thornton" as one entity, is the natural next
  experiment.
- **Hand-inspecting 30 misses** to estimate the redundancy correction
  factor PLAN.md calls for (a "miss" by this proxy may still be a
  perfectly good supporting passage phrased differently) — not done given
  the time budget; flagged as a known gap rather than skipped silently.
- Dense retrieval, hybrid RRF fusion, and reranking (PLAN.md Day 3 items
  2–3) — explicitly deferred to a later session per the time-budget
  decision in `PLAN.md`/`DEVLOG.md`.
