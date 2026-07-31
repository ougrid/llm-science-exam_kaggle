# Session handoff — 2026-07-31

State at end of session, what was learned, and what is worth doing next. Written
for whoever picks this up (including a future me with no memory of today).

## Current numbers

| | Public LB | Private LB |
|---|---|---|
| random baseline (dummy `"A B C"`) | 0.375156 | 0.356882 |
| single-index retrieval + public reader | 0.761131 | 0.747994 |
| **dual-index RRF fusion + public reader** ← current best | **0.762796** | **0.755183** |

Local, all on the **same full 1,500-row T1** (this matters — see "Sample drift"):

| | MAP@3 | 95% CI |
|---|---|---|
| random baseline | 0.3667 | analytic |
| my reader (`deberta-v3-base`, measured recipe) | 0.6906 | [0.6714, 0.7090] |
| known-good public reader (ceiling) | 0.7793 | [0.7628, 0.7958] |
| **reader-attributable loss** (paired) | **+0.0888** | [+0.0707, +0.1063] |

Provenance: corpus, chunking, both BM25 indexes, query construction, the RRF
fusion, and the whole offline pipeline are ours. **The reader in the submission is
a public checkpoint** (`mgoksu/llm-science-run-context-2`), which the rules permit
and `CREDITS.md` documents. It is not evidence of a model we trained.

## The two bugs that cost three days

Every training run in this repo pinned `train_loss` at exactly `ln(5) = 1.6094`
while gradients, optimizer, labels, context, and input format were all measurably
healthy. Six hypotheses, four wrong — all four blamed the data. The real causes
were **two borrowed constants that share one signature**:

1. **fp16 parameters.** `transformers` 5.x makes `from_pretrained` follow the
   *checkpoint's* stored dtype, and both `deberta-v3-base` and `-large` ship fp16.
   So AdamW updated half-precision weights in place. At lr=2e-5 an update is ~1.3
   ULP near a weight of 0.03 and **exactly zero for any weight ≥ 0.1** (which
   includes LayerNorm weights near 1.0). Hidden by `transformers>=4.46` with no
   upper bound.
2. **`lr=2e-5` × `freeze 18/24`.** Inherited from cdeotte part 2, which used it on
   ~60k rows with 5.6× more trainable parameters. A 2×2 on the full pool shows
   **exactly one corner of four learns** — the frozen lower layers are what make
   the higher LR usable, so both single-knob changes fail and the fully-unfrozen
   high-LR corner is *worse* than the recipe we started with.

Because both produce an identical outward symptom, fixing the first changed nothing
observable and read as "my diagnosis was wrong". It wasn't; it was insufficient.
Verify a fix against the **mechanism** (16-row overfit: 1.5687 fp16 vs 0.1013
fp32), not against the end-to-end symptom clearing.

**`LR = 2e-5  # cdeotte part 2's value` is the detail worth remembering.** A bare
magic number invites scrutiny; the same number with a provenance comment reads as
already-justified and deflected every re-read.

## Score levers: one win, four dead ends

All measured on the clean gold 200 with paired bootstrap (identical rows per arm).

| lever | verdict |
|---|---|
| **dual-index RRF fusion** | **SHIPPED.** +0.0193 recall@5 [+0.0053,+0.0327] resolved on 1,500 T1 rows; +0.0133 MAP@3 on gold 200 (unresolved at n=200); +0.0072 private LB |
| longer context (768/1024/1280) | dead — 3/3 arms negative; the checkpoint was fine-tuned at 512 and never saw longer positions |
| my trained reader alone | dead — 0.6906 vs the public reader's 0.7793 |
| ensemble mine + public | dead — rescues 15 rows, **breaks 57**; strictly worse, not uncorrelated |
| ensemble two strong publics | dead — best arm +0.0050 [-0.0075,+0.0175], unresolved |
| `deberta-v3-large` training | dead — does not leave ln(5) at three LRs, 282 steps at full LR |

**The dual-index mechanism is not what it looks like.** Either half of Wikipedia
scores *identically* alone (0.6207 vs 0.6220). It was never a coverage gap — the
corpus is redundant. The gain comes from fusing two independent views, which
promotes evidence each index alone ranked just below the top-5 cutoff. That is also
why RRF beat raw score-pooling: BM25's IDF is corpus-specific, so scores from
disjoint indexes are not comparable and rank-only fusion is the right primitive.

It also turned a hard constraint into an advantage. A single 3.02M-chunk index does
**not** build in 15 GB (MemoryError while tokenising at both 2.4M and 2.0M,
measured). Two half-indexes (1.60M + 1.42M) build easily, give the same coverage,
*and* add the fusion benefit.

## Sample drift — the methodological trap to not repeat

Three separate figures moved 0.018–0.032 today purely from *which* rows were
sampled, and **in opposite directions**:

- my reader: 0.6693 (500 rows, seed 42) vs **0.6906** (all 1,500)
- the ceiling: 0.7970 (500 rows, seed 0) vs **0.7793** (all 1,500)

The old attribution figure of **0.4130** subtracted across two different samples, so
it compounded both errors instead of cancelling them — *and* the reader it measured
was doubly broken. The honest value is **0.0888**, and 79% of the old number was
bugs. `PLAN.md` predicted this drift magnitude (±0.032 at n=500) before anything was
built. Writing it down did not prevent making the error; **paired designs on
identical rows** did. Use `scripts/attribution_paired_full_t1.py` as the pattern.

## Prediction calibration

| submission | predicted | actual public | miss |
|---|---|---|---|
| single-index | ~0.86 (from local gold 200) | 0.761131 | **0.086** |
| dual-index | 0.770, range 0.755–0.785 | 0.762796 | **0.007**, inside range |

One change fixed it: anchor on the **actual leaderboard number plus a measured
delta**, never on a 200-row local estimate. Also note the private split moved
+0.0072 against a predicted +0.009 — nearly exact, on the larger and less noisy
split. Pre-register every LB prediction in `experiments/lb_log.csv` *with* revert
and escalate triggers before submitting.

## What I would do next, in order

1. **Nothing, and bank the artifact.** Five levers tested, four dead with
   mechanisms. 0.762796 is close to what this pipeline yields with a borrowed
   reader. The interview material is strong: an honest attribution table, a
   two-bug diagnosis with an unusually rigorous falsification trail, and one
   measured score improvement. This is a legitimate stopping point.
2. **Third index view / reranked wider pool** (~2 h). The fusion win suggests the
   lever has more in it — but the escalate trigger (LB > 0.785) did *not* fire, so
   expect small. Retrieve top-20 from each half, rerank the fused pool to 5.
3. **Retrain the reader on the full 39,249-row pool** (~3 h local, zero quota). The
   4,586-row restriction came from falsified hypothesis 4; there is no measured
   reason for it. `scripts/train_reader_fixed_local.py --train-file
   data/train_pool_own_context_general_big.parquet --epochs 1
   --out-suffix=-bigpool` and then `scripts/compare_pool_size_paired.py`, whose
   verdict thresholds are pre-registered. This raises artifact quality, **not
   score** — even a good result stays below the public reader's 0.7793.
4. **Do NOT retry `deberta-v3-large`** (task #21) unless #3 shows a large gain.
   Its ceiling is below the public reader, so it cannot improve the submission.

## Gotchas that cost real time today

- **`ps -eo pid,etime,rss,cmd | grep <script>` is the liveness check.**
  `pgrep -af <pat> | grep -v pgrep` returned a false negative on a plainly-running
  training process; acting on it led to killing a healthy run 375 steps in. Also
  check `date -r <logfile>` against `date` — a log written seconds ago is proof of
  progress. **Never take a destructive action on a stale read.**
- **A performance guard must reproduce the config it guards.**
  `probe_training_speed()` optimised all 184.4M params while the run optimised
  22.2M, so the probe OOM'd where production was fine. Now takes
  `n_frozen_layers` / `freeze_embeddings` / `autocast_dtype`.
- **Run big-memory builds under `ulimit -v`.** The unguarded full-index build let
  the OOM killer take the *terminal*. Note `ulimit -v` caps virtual address space,
  which numpy over-reserves, so it trips earlier than RSS suggests.
- **On WSL2, exceeding VRAM raises `RuntimeError: CUDA driver error: device not
  ready`, not `torch.OutOfMemoryError`** — so `except torch.OutOfMemoryError`
  fallbacks (including the submission's CPU fallback) will not fire.
- **`kaggle kernels output` only serves FINISHED runs.** For a running kernel use
  `kaggle kernels logs -f <slug>`. A monitor built on the former is silent by
  design and looks identical to "no progress yet".
- **Kaggle kernel slugs derive from the title**, so `title` must match `id` or the
  push 409s. Submitting a Code Competition requires clicking **Submit to
  Competition** in the browser; the CLI 400s on file upload.

## Key files

- `README.md` — the 15-minute interview script; headline, results, attribution
- `DEVLOG.md` — full narrative, ~1,850 lines, every wrong turn preserved in order
- `CLAUDE.md` — engineering rules; the last several were earned today
- `experiments/log.csv` — every measurement with CIs and a one-line hypothesis
- `experiments/lb_log.csv` — LB scores against pre-registered predictions
- `scripts/hypothesis_gate.py` — pre-flight gate; **check 0 audits training dtypes**
- `scripts/attribution_paired_full_t1.py` — the paired-attribution pattern
- `scripts/compare_dual_index_recall.py` — the fusion measurement that shipped
- `notebooks/kaggle/day3-submission/` — the live submission (v4, dual-index RRF)

Working tree clean, all pushed. 62 tests passing. Last commit `d99daad`.
