"""SCORE LEVER: does ensembling my trained reader with the public one beat it alone?

The situation this exists to exploit. On the same 1,500 T1 rows: public reader
0.7793 [0.7628,0.7958], my trained reader 0.6906 [0.6714,0.7090]. My reader is
strictly worse, so it cannot improve the submission by replacing anything. Its only
route to a higher score is contributing signal the public reader lacks -- which
requires the two to make DIFFERENT mistakes, not merely for one to be good.

That is the whole question here, and it is an empirical one. Two readers can be 9
points apart and still ensemble well (uncorrelated errors) or ensemble to nothing
(the weaker one is a strict subset of the stronger). Nothing in the scores above
tells us which. What does tell us is the disagreement structure, which this prints.

Fusion methods, both tried because they fail differently:
  * Z-NORMALISED LOGIT AVERAGE. Raw logits are not comparable across independently
    trained models -- different scales, different calibration -- so averaging them
    directly is dominated by whichever model happens to be more confident. Z-scoring
    per row per model fixes the scale but keeps the margins.
  * RECIPROCAL RANK FUSION (k=60). Throws magnitude away entirely and uses only
    rank order. More robust when calibration differs wildly; loses information when
    it doesn't. src/llmsci/retrieve/fuse.py already implements RRF and is tested.

MAP@3 depends only on rank order, so any monotone rescaling of a SINGLE model's
scores is a no-op -- calibration matters here precisely because we are combining
two models, which is the one case where it stops being irrelevant.

Weights are swept coarsely rather than tuned finely: selecting a weight on 200 rows
would buy best-of-N optimism worth more than the effect (PLAN.md -- 200 rows
resolves ~0.04 at best). A weight only ships if it wins at a coarse grid AND its
paired CI against the public reader alone excludes 0.

Measured on the clean official 200 with our retrieval, the tier the submitted
0.8592 came from. Paired bootstrap throughout: every arm scores the SAME rows.

Run: python scripts/ensemble_readers_gold.py
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

DATA = Path("data")
GOLD_CTX = DATA / "gold_own_context_general.parquet"
PUBLIC = Path("reference_reproduction/models/mgoksu-run-context-2")
MINE = DATA / "checkpoints" / "base-fixed-lr1e-4"
OPTIONS = ["A", "B", "C", "D", "E"]
BASELINE = random_baseline_map_at_k()
RRF_K = 60
WEIGHTS = [0.0, 0.2, 0.3, 0.4, 0.5]  # weight on MY reader; 0.0 == public alone


def logits_for(ckpt: Path, df, layout: str, device) -> np.ndarray:
    from transformers import AutoModelForMultipleChoice, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(ckpt)
    model = AutoModelForMultipleChoice.from_pretrained(ckpt, dtype=torch.float32).to(device).eval()

    class DS(Dataset):
        def __len__(self):
            return len(df)

        def __getitem__(self, i):
            r = df.iloc[i]
            if layout == "public":   # what this checkpoint was fine-tuned on
                first = [f"{str(r['context'])[:1750]} #### {r['prompt']}"] * 5
                second = [str(r[c]) for c in OPTIONS]
                maxlen = 512
            else:                    # our training layout
                first = [str(r["context"])[:8000]] * 5
                second = [f"{r['prompt']} {r[c]}" for c in OPTIONS]
                maxlen = 384
            return dict(tok(first, second, truncation="only_first", max_length=maxlen))

    loader = DataLoader(DS(), batch_size=2, shuffle=False,
                        collate_fn=DataCollatorForMultipleChoice(tok))
    out = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16,
                                         enabled=(device.type == "cuda")):
        for b in loader:
            b.pop("labels", None)
            out.append(model(**{k: v.to(device) for k, v in b.items()}).logits.float().cpu().numpy())
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return np.concatenate(out, axis=0)


def zscore(x: np.ndarray) -> np.ndarray:
    mu = x.mean(axis=1, keepdims=True)
    sd = x.std(axis=1, keepdims=True)
    return (x - mu) / np.where(sd == 0, 1.0, sd)


def rrf_scores(*logit_sets: np.ndarray, k: int = RRF_K) -> np.ndarray:
    """Higher is better. Rank 0 = best option within a row."""
    total = np.zeros_like(logit_sets[0])
    for lg in logit_sets:
        order = np.argsort(-lg, axis=1)
        ranks = np.empty_like(order)
        rows = np.arange(lg.shape[0])[:, None]
        ranks[rows, order] = np.arange(lg.shape[1])[None, :]
        total += 1.0 / (k + ranks + 1)
    return total


def ap(scores: np.ndarray, answers) -> np.ndarray:
    return average_precision_scores(answers, logits_to_ranked_labels(scores, k=3), k=3)


def main() -> None:
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--mine", default=str(MINE))
    args = ap_.parse_args()
    mine = Path(args.mine)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        cap_memory_fraction(0.975)
    if not GOLD_CTX.exists():
        print(f"{GOLD_CTX} missing -- run scripts/sweep_context_length_gold.py first")
        return
    df = pd.read_parquet(GOLD_CTX)
    answers = df["answer"].tolist()
    print(f"clean gold {len(df)} rows, our retrieval | baseline {BASELINE:.4f}")

    lg_pub = logits_for(PUBLIC, df, "public", device)
    lg_mine = logits_for(mine, df, "ours", device)

    s_pub = ap(lg_pub, answers)
    s_mine = ap(lg_mine, answers)
    for tag, s in (("public reader alone", s_pub), ("my reader alone", s_mine)):
        m, lo, hi = bootstrap_ci(s, n_resamples=10_000, seed=0)
        print(f"  {tag:22s} MAP@3 {m:.4f} [{lo:.4f},{hi:.4f}]")

    # The precondition for any ensemble gain. If the weaker model is never right
    # where the stronger is wrong, there is no signal to add and no fusion method
    # can invent one.
    top_pub = lg_pub.argmax(1)
    top_mine = lg_mine.argmax(1)
    gold = np.array([OPTIONS.index(a) for a in answers])
    disagree = float((top_pub != top_mine).mean())
    rescue = int(((top_pub != gold) & (top_mine == gold)).sum())
    break_ = int(((top_pub == gold) & (top_mine != gold)).sum())
    print(f"\n  top-1 disagreement: {disagree:.3f} of rows")
    print(f"  rows my reader gets right and public gets WRONG: {rescue}")
    print(f"  rows public gets right and mine gets wrong:      {break_}")
    if rescue == 0:
        print("  => no complementary signal at all; an ensemble cannot help. Stop here.")

    zp, zm = zscore(lg_pub), zscore(lg_mine)
    results = {}
    print("\n  z-normalised logit average (weight on MY reader):")
    for w in WEIGHTS:
        s = ap((1 - w) * zp + w * zm, answers)
        m, lo, hi = bootstrap_ci(s, n_resamples=10_000, seed=0)
        results[f"zavg_w{w}"] = s
        print(f"    w={w:<4} MAP@3 {m:.4f} [{lo:.4f},{hi:.4f}]")
    s_rrf = ap(rrf_scores(lg_pub, lg_mine), answers)
    results["rrf"] = s_rrf
    m, lo, hi = bootstrap_ci(s_rrf, n_resamples=10_000, seed=0)
    print(f"  RRF(k={RRF_K}):        MAP@3 {m:.4f} [{lo:.4f},{hi:.4f}]")

    print("\n  PAIRED vs public reader alone (same rows):")
    winners = []
    for tag, s in results.items():
        d, lo, hi = paired_bootstrap(s_pub, s, n_resamples=10_000, seed=0)  # mean(s - public)
        mark = "RESOLVED" if (lo > 0 or hi < 0) else "not resolved by 200 rows"
        print(f"    {tag:12s} delta {d:+.4f} [{lo:+.4f},{hi:+.4f}]  {mark}")
        if d > 0 and lo > 0:
            winners.append((tag, d))

    print()
    if winners:
        tag, d = max(winners, key=lambda x: x[1])
        print(f"SHIP IT: {tag} beats the public reader alone by {d:+.4f}, CI excludes 0.")
        print("  Add my checkpoint as a second Kaggle dataset and fuse at inference.")
        print("  Cost: one extra forward pass per row -- the reader was never the")
        print("  bottleneck (~8 min/model per 4,000 rows on a T4), so the 9 h limit holds.")
    else:
        print("NO ensemble arm beats the public reader with a CI excluding 0 on 200 rows.")
        print("  Do not ship on an unresolved delta. Note 200 rows resolves ~0.04 at best,")
        print("  so a real +0.02 would hide here -- report as unresolved, not as absent.")
    Path("reports").mkdir(exist_ok=True)
    Path("reports/ensemble_gold.json").write_text(json.dumps({
        "public_alone": list(bootstrap_ci(s_pub, n_resamples=10_000, seed=0)),
        "mine_alone": list(bootstrap_ci(s_mine, n_resamples=10_000, seed=0)),
        "top1_disagreement": disagree, "mine_rescues": rescue, "mine_breaks": break_,
        "arms": {k: list(bootstrap_ci(v, n_resamples=10_000, seed=0)) for k, v in results.items()},
    }, indent=2))


if __name__ == "__main__":
    main()
