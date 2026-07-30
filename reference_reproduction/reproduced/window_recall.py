"""Does reranking put better evidence INSIDE the reader's 256-token window?

This tests the modernization's causal mechanism directly, with no reader and no
training, which makes it both far cheaper and far cleaner than inferring the
mechanism from a MAP@3 delta.

The claim being tested (from ../NOTES.md): T1's retrieved context is a median
1,067 tokens against MAX_INPUT=256 with truncation='only_first', so ~3/4 of it
is discarded, and MiniLM's ordering alone decides what survives. If a 2026
cross-encoder orders the sentences better, then the answer-supporting sentence
should land inside the surviving 256 tokens MORE OFTEN -- and that, not
anything else, is the channel through which the reranker could help this reader.

Metric: answer-support hit rate on the truncated window, reusing
`llmsci.retrieve.eval`'s own definitions (`distinctive_keywords`,
`is_answer_support_hit`) so the number is directly comparable to the main
pipeline's Proxy B retrieval numbers rather than a private reinvention.

Paired per-row, same 1,500 T1 rows, same truncation the reader actually applies.

Usage: python window_recall.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from common import MAX_INPUT, MODEL_NAME
from llmsci.metrics import bootstrap_ci, minimum_detectable_effect, paired_bootstrap
from llmsci.retrieve.eval import distinctive_keywords, is_answer_support_hit

DATA = Path(__file__).resolve().parent.parent / "data"
RESULTS = Path(__file__).resolve().parent.parent / "results"


def truncated_window(tokenizer, context: str, budget: int) -> str:
    """The part of `context` that survives truncation='only_first' at max_length=budget.

    The reader's first segment is "[CLS] " + context and its second segment is
    " #### prompt [SEP] option [SEP]"; only_first cuts the first segment so the
    pair fits `budget`. The second segment's length therefore sets how much
    context survives, so it is measured per row rather than assumed.
    """
    return context  # replaced per-row below; kept for clarity of intent


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    df = pd.read_parquet(DATA / "t1_eval_cdeotte_ctx.parquet")
    if "context_reranked" not in df.columns:
        raise KeyError("run rerank.py first")
    print(f"T1 rows: {len(df)}, budget {MAX_INPUT} tokens\n")

    hits = {"minilm": [], "reranked": []}
    kept_tokens = {"minilm": [], "reranked": []}
    n_no_keywords = 0

    for row in df.itertuples(index=False):
        r = row._asdict()
        kws = distinctive_keywords(pd.Series(r))
        if not kws:
            n_no_keywords += 1
        # Worst case over the five options: the longest second segment leaves the
        # least room for context. Using the correct option's segment is what the
        # answer-bearing forward pass actually sees.
        second = f" #### {r['prompt']} [SEP] {r[r['answer']]} [SEP]"
        n_second = len(tokenizer(second, add_special_tokens=False)["input_ids"])
        ctx_budget = max(MAX_INPUT - n_second, 0)

        for key, col in [("minilm", "context"), ("reranked", "context_reranked")]:
            ids = tokenizer(f"[CLS] {r[col]}", add_special_tokens=False)["input_ids"]
            kept_ids = ids[:ctx_budget]
            kept_text = tokenizer.decode(kept_ids)
            kept_tokens[key].append(len(kept_ids))
            hits[key].append(1.0 if is_answer_support_hit(kept_text, kws) else 0.0)

    a = np.array(hits["minilm"])
    b = np.array(hits["reranked"])
    am, alo, ahi = bootstrap_ci(a, n_resamples=10_000, seed=0)
    bm, blo, bhi = bootstrap_ci(b, n_resamples=10_000, seed=0)
    dm, dlo, dhi = paired_bootstrap(a, b, n_resamples=10_000, seed=0)
    mde = minimum_detectable_effect(a, b)
    resolved = (dlo > 0) or (dhi < 0)

    print(f"rows with no distinctive keyword (metric undefined, counted as miss): {n_no_keywords}")
    print(f"context tokens surviving truncation: mean {np.mean(kept_tokens['minilm']):.0f} "
          f"of a median 1067-token context\n")
    print(f"answer-support hit rate INSIDE the {MAX_INPUT}-token window (n={len(a)}):")
    print(f"  MiniLM order (as shipped) : {am:.4f}  95% CI [{alo:.4f}, {ahi:.4f}]")
    print(f"  gte-reranker-modernbert   : {bm:.4f}  95% CI [{blo:.4f}, {bhi:.4f}]")
    print(f"\n  paired delta: {dm:+.4f}  95% CI [{dlo:+.4f}, {dhi:+.4f}]")
    print(f"  rows that changed: {(a != b).mean():.1%}   MDE at this n: +/-{mde:.4f}")
    print(f"  verdict: {'RESOLVED' if resolved else 'NOT RESOLVED BY THIS EVAL (CI includes 0)'}")

    # Full-context hit rate is the ceiling reranking works against: reordering
    # cannot add evidence, only move it inside the window.
    full = np.array([
        1.0 if is_answer_support_hit(str(r._asdict()["context"]),
                                     distinctive_keywords(pd.Series(r._asdict()))) else 0.0
        for r in df.itertuples(index=False)
    ])
    fm, flo, fhi = bootstrap_ci(full, n_resamples=10_000, seed=0)
    print(f"\n  full untruncated context : {fm:.4f}  95% CI [{flo:.4f}, {fhi:.4f}]  "
          f"<- ceiling for any reordering")
    print(f"  headroom truncation costs: {fm - am:.4f} (MiniLM) / {fm - bm:.4f} (reranked)")

    with open(RESULTS / "window_recall.json", "w") as f:
        json.dump({
            "n": len(a), "budget_tokens": MAX_INPUT,
            "minilm_hit_rate": am, "minilm_ci": [alo, ahi],
            "reranked_hit_rate": bm, "reranked_ci": [blo, bhi],
            "paired_delta": dm, "paired_delta_ci": [dlo, dhi],
            "rows_changed": float((a != b).mean()), "mde": mde, "resolved": bool(resolved),
            "full_context_hit_rate": fm, "full_context_ci": [flo, fhi],
            "mean_context_tokens_kept": float(np.mean(kept_tokens["minilm"])),
            "rows_without_keywords": n_no_keywords,
        }, f, indent=2)
    print(f"\nwrote {RESULTS/'window_recall.json'}")


if __name__ == "__main__":
    main()
