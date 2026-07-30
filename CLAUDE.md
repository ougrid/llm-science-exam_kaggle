# CLAUDE.md

Engineering conventions for this repo. `PLAN.md` is the technical plan (build order, validation design, retrieval/reader architecture, the 2026 stack). This file is the *how we work* layer on top of it — read both.

## What this project actually is

A reproduction study of Kaggle's closed "LLM Science Exam" competition, built as an interview-grade portfolio artifact. The thesis: retrieval quality beats parameter count on this task, and the artifact's value comes from *proving that with measurement*, not from a leaderboard number. See `PLAN.md` for the full argument.

## Non-negotiable rules (violating these silently invalidates the project)

These aren't style preferences — they're the difference between a defensible result and a number nobody should trust.

- **Never train on the official 200 gold rows** (`data/holdout_gold.csv`). They are the held-out test set, full stop. If a change touches training data, verify the gold set isn't in it before running anything.
- **No naked MAP@3 numbers.** Every reported score gets a 95% bootstrap CI. A number without a CI doesn't go in a report, a commit message, or a comment.
- **Comparisons use paired bootstrap, not two unpaired CIs.** Two configs run on the same frozen eval set, per-row AP differenced, then bootstrapped. This is in `src/llmsci/metrics.py` — use it, don't reinvent it inline.
- **Cap gold-set evaluations at ~8 for the whole project**, each logged in `experiments/log.csv` with a date and a reason. The gold set is spent capital, not a free check.
- **Retrieval training contexts must come from the same retriever used at test time.** Training on cleaner context than inference sees is the most common RAG bug in this codebase's design and it will not be caught by any test — watch for it by hand.
- **Numbers come from measurement, not intuition.** Every threshold, hyperparameter default, or "expected ~X" claim in a docstring or report must trace to a run in `experiments/log.csv`. Don't invent one because it "sounds about right."
- **Mark noise-band results as noise-band results.** If a paired-bootstrap CI on a gain includes 0, the report says "not resolved by my eval" — it does not get quietly rounded up to a win.

## Engineering discipline

- **Small, reversible steps.** One logical change per commit, committed once it works. History should be bisectable — each diff matches its message.
- **Verify, don't assume.** A metric, a shape assertion, a retrieval score, a submission format — check it against a real run before calling it done. "It should work" is not verification.
- **Report honestly.** If a check fails, a phase was skipped, or a number came out worse than expected, say so with the actual output. Never claim a result without having looked at it.
- **No scope creep.** Fix incidental issues only when they block the current phase or are a one-line cleanup; keep those in their own commit, separate from the phase's main work.
- **Respect existing contracts.** Match `PLAN.md`'s repo layout, config schema, and column names exactly rather than inventing better ones mid-build. If a change to the plan is genuinely warranted, update `PLAN.md` in the same commit and say so in the message.
- **No AI-slop tells.** No emoji headers, no "comprehensive," no restating the task back at the reader, no tables that carry one fact per row.
- **We do not reward verbosity.** A concise, well-reasoned notebook cell or report section beats a lengthy generic one — this is doubly true here since the whole point is a 15-minute interview walkthrough, not an exhaustive dump.
- **A monitor's silence is not evidence a job is still running — check the process directly.** A log-tailing watcher only fires on lines matching its grep filter; if the filter doesn't match the script's actual print format, the watcher goes silent forever and looks identical to "still waiting," even after the job finished or died minutes ago. This has already produced a wrong "still running" claim once (see DEVLOG.md). Before reporting status on a background job, especially after any gap, re-verify with `ps` and/or by reading the log file directly rather than trusting a monitor's absence of notifications. This is the same "coverage — silence is not success" trap that applies to writing the filter in the first place, just as easy to fall into when *reading* one.

## Git

- **Only commit when explicitly asked.** Never force-push, never rewrite published history on `main`, never `--no-verify`.
- **Conventional Commits.** `<type>(<scope>): <imperative present-tense summary>`, lowercase, no trailing period. Common types here: `feat`, `fix`, `data`, `eval`, `docs`, `chore`. Scope is a noun matching the touched area (`retrieval`, `reader`, `metrics`, `kaggle`, `corpus`). Body explains *why*, not a restated diff — e.g. what the run showed, not "changed lr from X to Y."
- Tag the end of each core-track day (`day1-baseline`, `day2-retrieval`, `day3-attribution`, `day4-submission`) once its deliverable is committed — the tags double as the reproducibility trail referenced in the README.

## Stack

- **`uv`-managed**, not raw `pip`/`venv` — the system `python3-venv` is broken on the dev box, and `uv` sidesteps it entirely. Pin Python **3.11** to match Kaggle's base image.
- **Local (RTX 5050, Blackwell sm_120):** torch from the cu128 index, bf16 for training. Verify `torch.cuda.get_device_capability() == (12, 0)` *and* a real fp16 matmul + backward before trusting the environment — `is_available()` returning `True` on a mismatched wheel is a known trap that only fails at first kernel launch.
- **WSL2 GPU memory: cap it and probe it, every time, before the real run starts.** WSL2's CUDA driver silently backs allocations past the physical 8.55 GB VRAM limit with system RAM ("shared GPU memory"), which runs 15-25x slower with zero error, warning, or unusual `nvidia-smi` reading — a training run once cost 5 hours before this was caught by hand (full story in `DEVLOG.md`, "WSL2 shared-memory trap"). `nvidia-smi` showing GPU utilization is *not* evidence a job is healthy; it looks identical whether the job is computing against VRAM or against slow PCIe-backed system RAM. Every GPU training/inference script must, before loading the real model:
  1. Call `llmsci.gpu_guard.cap_memory_fraction()` — turns silent oversubscription into an immediate, loud CUDA OOM.
  2. Call `llmsci.gpu_guard.probe_training_speed()` (or `assert_step_speed` for non-training GPU work) with a `max_ms_per_step` from a real measurement on this hardware for this exact batch/seq-len config — turns any *other* silent slowdown (thermal throttle, CPU contention, a driver hiccup) into an immediate failure instead of hours of undetected crawling. Don't guess the threshold; benchmark it once and hardcode the number with a comment citing the measurement.
  Long-running loops (training steps, batched inference) must also print periodic elapsed/rate/ETA — a silent multi-minute gap is indistinguishable from a hang and has to be re-verified by hand every time otherwise. To eyeball an already-running job, `python scripts/gpu_health_check.py` reports memory headroom against the true card size plus temperature/power (the fallback path is PCIe-bound, so it draws far less power than the reported utilization would suggest for real compute).
- **Kaggle (2×T4, Turing):** fp16 only — no bf16, no FlashAttention-2. DeBERTa-v3's `layer_norm_eps=1e-7` is a known fp16 NaN source on T4; keep LayerNorm in fp32 if it appears.
- **Never build a global dense index in local RAM** (15 GiB total) — stream corpus data with `pyarrow.parquet.ParquetFile.iter_batches()`, never `pd.concat` the full corpus.
- **No ANN index.** The corpus is small enough for exact `IndexFlatIP` / `torch.matmul` — see `PLAN.md` for the arithmetic. Don't add FAISS IVF/HNSW tuning; there's no recall to gain from it here.
- Model/embedder query-passage prefixes vary by model and silently degrade recall if wrong — use `encode_query()`/`encode_document()` (sentence-transformers ≥5.0) rather than hand-writing prefix strings.

## Repo layout

Follow `PLAN.md`'s "Repo layout" section exactly. In short: `src/llmsci/` for all reusable code, `notebooks/` for the narrative and the Kaggle submission notebooks (which import from `src/`, never duplicate its logic), `configs/` one YAML per ablation row, `experiments/log.csv` + `lb_log.csv` for every recorded run, `reports/` for the artifacts an interviewer actually sees, `tests/` covering metrics, the collator shapes, leakage, and submission format.

Large artifacts (checkpoints, corpus, wheels) never go in git — they go to Kaggle Datasets per `src/llmsci/kaggle/push_dataset.py`. If you're about to `git add` something over a few MB, stop and check whether it belongs in `.gitignore` instead.
