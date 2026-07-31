"""Does the resolved +0.0193 recall gain from dual-index fusion convert to MAP@3?

Established by scripts/compare_dual_index_recall.py on all 1,500 T1 rows: fusing
the two corpus halves with RRF(k=60) beats the shipped single index by +0.0193
[+0.0053,+0.0327] answer-support recall@5 -- resolved, CI excludes 0. That is the
first resolved improvement found today, and it needed the full 1,500 rows to see;
at n=600 the same effect sat at +0.0167 [-0.0050,+0.0400].

But recall is UPSTREAM of the score. More answer support in the top-5 only helps if
the reader converts it, and the conversion is sub-linear -- distractors also enter
with the extra context. So this measures MAP@3 directly, with the public reader that
the submission actually uses, single-index context vs RRF-fused context, paired on
identical rows.

WHY THE GOLD 200 AND NOT T1, despite T1 being 7.5x larger and better powered. The
public reader scores 0.9170 on T1, which is 100% contaminated -- that checkpoint's
training pool overlaps T1's source articles. Using T1 here would measure
memorisation, not retrieval. The clean official 200 is the only honest tier for this
reader, and its +-0.04 CI half-width means a realistic +0.01 MAP@3 CANNOT resolve
here. That is a known limit, not a surprise.

So the decision rule, stated before running, because "unresolved" is the likely
outcome and it must not be reinterpreted afterwards:

  * MAP@3 delta clearly NEGATIVE -> do not ship, regardless of the recall win. The
    extra context is crowding out good evidence.
  * delta >= 0 but unresolved -> shipping is defensible on the RESOLVED upstream
    recall gain plus a non-negative conversion. The change touches retrieval only,
    needs no reader change, and the hidden test's ~4,000 rows would show a true
    +0.01 that 200 rows cannot. Say plainly that the MAP@3 gain is unresolved.
  * delta resolved positive -> ship without qualification.

Run: python scripts/eval_dual_index_map3_gold.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from llmsci.gpu_guard import cap_memory_fraction
from llmsci.metrics import (
    average_precision_scores,
    bootstrap_ci,
    paired_bootstrap,
    random_baseline_map_at_k,
)
from llmsci.reader.mc import DataCollatorForMultipleChoice, logits_to_ranked_labels
from llmsci.retrieve.sparse import BM25Index, build_query

DATA = Path("data")
GOLD = DATA / "holdout_gold.csv"
IDX_A = DATA / "bm25_index_general"
IDX_B = DATA / "bm25_index_general_oddhalf"
PUBLIC = Path("reference_reproduction/models/mgoksu-run-context-2")
OPTIONS = ["A", "B", "C", "D", "E"]
BASELINE = random_baseline_map_at_k()
K = 5
RRF_K = 60
CONTEXT_CHAR_CLIP = 1750   # the submitted setting; the sweep found longer does not help
MAX_LENGTH = 512


def retrieve(index_dir: Path, queries: list[str]) -> list[list[str]]:
    chunks = pd.read_parquet(index_dir / "chunk_texts.parquet", columns=["text"])
    texts = chunks["text"].tolist()
    del chunks
    index = BM25Index.load(index_dir, texts)
    print(f"  {index_dir.name}: {len(texts):,} chunks", flush=True)
    out = [[texts[i] for i, _ in res] for res in index.search_batch(queries, k=K)]
    del index, texts
    return out


def rrf_fuse(a: list[str], b: list[str]) -> list[str]:
    scored: dict[str, float] = {}
    for rank, txt in enumerate(a):
        scored[txt] = scored.get(txt, 0.0) + 1.0 / (RRF_K + rank + 1)
    for rank, txt in enumerate(b):
        scored[txt] = scored.get(txt, 0.0) + 1.0 / (RRF_K + rank + 1)
    return [t for t, _ in sorted(scored.items(), key=lambda kv: -kv[1])[:K]]


def score_reader(df: pd.DataFrame, device) -> np.ndarray:
    from transformers import AutoModelForMultipleChoice, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(PUBLIC)
    model = AutoModelForMultipleChoice.from_pretrained(PUBLIC).to(device).eval()

    class DS(Dataset):
        def __len__(self):
            return len(df)

        def __getitem__(self, i):
            r = df.iloc[i]
            first = f"{str(r['context'])[:CONTEXT_CHAR_CLIP]} #### {r['prompt']}"
            return dict(tok([first] * 5, [str(r[c]) for c in OPTIONS],
                            truncation="only_first", max_length=MAX_LENGTH))

    loader = DataLoader(DS(), batch_size=2, shuffle=False,
                        collate_fn=DataCollatorForMultipleChoice(tok))
    out = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16,
                                         enabled=(device.type == "cuda")):
        for b in loader:
            b.pop("labels", None)
            out.append(model(**{k: v.to(device) for k, v in b.items()}).logits.float().cpu().numpy())
    logits = np.concatenate(out, axis=0)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return average_precision_scores(df["answer"].tolist(),
                                    logits_to_ranked_labels(logits, k=3), k=3)


def main() -> None:
    argparse.ArgumentParser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        cap_memory_fraction(0.975)
    if not IDX_B.exists():
        print(f"{IDX_B} missing -- build the complement half first")
        return

    gold = pd.read_csv(GOLD)
    queries = [build_query(r["prompt"], [r[c] for c in OPTIONS]) for _, r in gold.iterrows()]
    print(f"clean gold {len(gold)} rows | baseline {BASELINE:.4f} | reader {PUBLIC.name}")

    # Sequential: never both indexes resident (that co-residency is the original OOM).
    a = retrieve(IDX_A, queries)
    b = retrieve(IDX_B, queries)

    single = gold.copy()
    single["context"] = [" ".join(x) for x in a]
    fused = gold.copy()
    fused["context"] = [" ".join(rrf_fuse(a[i], b[i])) for i in range(len(gold))]
    changed = int(sum(1 for i in range(len(gold))
                      if fused["context"].iloc[i] != single["context"].iloc[i]))
    print(f"  context differs on {changed}/{len(gold)} rows", flush=True)

    s_single = score_reader(single, device)
    m1 = bootstrap_ci(s_single, n_resamples=10_000, seed=0)
    print(f"\n  single index (shipped) MAP@3 {m1[0]:.4f} [{m1[1]:.4f},{m1[2]:.4f}]")
    s_fused = score_reader(fused, device)
    m2 = bootstrap_ci(s_fused, n_resamples=10_000, seed=0)
    print(f"  dual-index RRF fusion  MAP@3 {m2[0]:.4f} [{m2[1]:.4f},{m2[2]:.4f}]")

    d, lo, hi = paired_bootstrap(s_single, s_fused, n_resamples=10_000, seed=0)  # fused - single
    resolved = lo > 0 or hi < 0
    print(f"\n  paired delta (fused - single): {d:+.4f} [{lo:+.4f},{hi:+.4f}]")
    print(f"  {'RESOLVED' if resolved else 'NOT resolved -- 200 rows cannot see ~0.01'}")
    print("\nDECISION (rule declared in the docstring before this ran)")
    if d < -0.005:
        print("  DO NOT SHIP. The fused context makes MAP@3 worse; the extra evidence is")
        print("  crowding out good chunks at k=5 despite higher answer-support recall.")
    elif resolved and d > 0:
        print(f"  SHIP WITHOUT QUALIFICATION: {d:+.4f} resolved on the clean gold 200.")
    else:
        print(f"  SHIPPING IS DEFENSIBLE: MAP@3 {d:+.4f} is non-negative and the UPSTREAM")
        print("  recall gain is resolved (+0.0193 [+0.0053,+0.0327] on 1,500 rows).")
        print("  Retrieval-only change, no reader change, and ~4,000 hidden rows would")
        print("  show a true +0.01 that 200 rows cannot. Report the MAP@3 gain as")
        print("  UNRESOLVED -- do not present it as a measured improvement.")
    Path("reports").mkdir(exist_ok=True)
    Path("reports/dual_index_map3_gold.json").write_text(json.dumps({
        "single_index": list(m1), "dual_index_rrf": list(m2),
        "paired_delta": [d, lo, hi], "resolved": bool(resolved),
        "rows_context_changed": changed, "n": len(gold),
        "upstream_recall_delta": [0.0193, 0.0053, 0.0327],
    }, indent=2))


if __name__ == "__main__":
    main()
