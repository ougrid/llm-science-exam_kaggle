"""Why is our own reader pinned at ln(5) while the same data scores 0.7970?

The gate (scripts/hypothesis_gate.py) cleared three suspects but left a hole I
only saw after reading the day4-src2-train log:

  * check 1 verified that each row's stored CONTEXT matches its own query.
  * check 3 verified that the EVAL file (t1_dev_own_context_general_big) is
    learnable -- a known-good reader scores 0.7970 [0.7687,0.8247] on it.
  * NOTHING ever verified the TRAIN file. Not its labels, not its context
    usefulness, not the input format the training script feeds it in.

That matters because the day-4 symptom is specifically a TRAINING-side failure:
train_loss sat at 1.6118-1.6173 against ln(5)=1.6094 for 1,000 optimizer steps
across 4 epochs of only 4,586 rows. A model with 77.2M trainable parameters
cannot fail to reduce TRAIN loss on 4,586 rows for four epochs unless either
gradients are not flowing or the labels carry no information about the inputs.
Eval being at baseline is downstream of that; the train loss is the tell.

Same instrument-inversion trick as the gate: use the known-good reader
(mgoksu/llm-science-run-context-2, 0.8600 on the clean gold 200 with our
context) to interrogate the DATA and the FORMAT, since it is the one component
we know works.

Three cells, each one variable away from the 0.7970 reference:

  A. eval file  + gate format      -> reproduces 0.7970 (control; must match)
  B. TRAIN file + gate format      -> are the TRAIN file's labels/context sound?
     If this lands near 0.3667 while A is 0.7970, the training data is the bug
     and no optimizer change can fix it.
  C. eval file  + TRAINING format  -> does our training script's input format
     (context as segment 1 capped at 8,000 chars, "prompt option" as segment 2,
     max_length 384) destroy the signal? Compared against A this isolates the
     format, on the set where we already know the answer should be ~0.80.

Cell C is confounded in one direction worth stating: this reader was fine-tuned
with the "context #### prompt" / option layout, so ANY layout change costs it
something. So C is an upper bound on format damage, not a clean estimate --
a small drop in C exonerates the format, a collapse in C indicts it.

Forward passes only. Runs on the local 8 GB card in a few minutes; no GPU quota.

Run: python scripts/diagnose_train_data_and_format.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from llmsci.gpu_guard import cap_memory_fraction
from llmsci.metrics import average_precision_scores, bootstrap_ci, random_baseline_map_at_k
from llmsci.reader.mc import DataCollatorForMultipleChoice, logits_to_ranked_labels

DATA = Path("data")
PUBLIC_MODEL = Path("reference_reproduction/models/mgoksu-run-context-2")
OPTIONS = ["A", "B", "C", "D", "E"]
BASELINE = random_baseline_map_at_k()

TRAIN_SRC2 = DATA / "train_pool_own_context_src2.parquet"
T1_CTX = DATA / "t1_dev_own_context_general_big.parquet"

N = 500
SEED = 0

# The format the gate / the 0.8600 submission uses.
GATE_FMT = dict(context_chars=1750, max_length=512, layout="ctx_hash_prompt")
# The format notebooks/kaggle/day4-src2-train/script.py trains with.
TRAIN_FMT = dict(context_chars=8000, max_length=384, layout="ctx_then_prompt_opt")


def encode(tok, row, fmt):
    ctx = str(row["context"])[: fmt["context_chars"]]
    if fmt["layout"] == "ctx_hash_prompt":
        first = [f"{ctx} #### {row['prompt']}"] * 5
        second = [str(row[c]) for c in OPTIONS]
    else:  # ctx_then_prompt_opt -- build_choice_texts() in the training script
        first = [ctx] * 5
        second = [f"{row['prompt']} {row[c]}" for c in OPTIONS]
    return dict(tok(first, second, truncation="only_first", max_length=fmt["max_length"]))


def score(model, tok, df, fmt, device, label):
    class DS(Dataset):
        def __len__(self):
            return len(df)

        def __getitem__(self, i):
            return encode(tok, df.iloc[i], fmt)

    loader = DataLoader(DS(), batch_size=2, shuffle=False,
                        collate_fn=DataCollatorForMultipleChoice(tok))
    out = []
    with torch.no_grad():
        for b in loader:
            b.pop("labels", None)
            out.append(model(**{k: v.to(device) for k, v in b.items()}).logits.float().cpu().numpy())
    logits = np.concatenate(out, axis=0)
    scores = average_precision_scores(df["answer"].tolist(), logits_to_ranked_labels(logits, k=3), k=3)
    m, lo, hi = bootstrap_ci(scores, n_resamples=10_000, seed=0)
    # A collapsed / uninformative model shows up as near-identical logits across
    # options, which MAP@3 alone cannot distinguish from confident-but-wrong.
    spread = float(np.mean(logits.max(axis=1) - logits.min(axis=1)))
    print(f"  {label:<44} MAP@3 {m:.4f} [{lo:.4f},{hi:.4f}]  logit spread {spread:.3f}", flush=True)
    return m


def truncation_report(tok, df, fmt, label):
    """How much of the retrieved context actually survives into the model?"""
    kept, total = [], []
    for _, r in df.head(100).iterrows():
        ctx = str(r["context"])[: fmt["context_chars"]]
        full = len(tok(ctx, add_special_tokens=False)["input_ids"])
        enc = encode(tok, r, fmt)
        # segment 1 length = tokens where token_type_ids == 0, minus specials
        tt = np.asarray(enc["token_type_ids"][0])
        kept.append(int((tt == 0).sum()) - 2)
        total.append(full)
    frac = float(np.sum(kept) / max(np.sum(total), 1))
    print(f"  {label:<44} context tokens kept {np.mean(kept):.0f}/{np.mean(total):.0f} "
          f"= {100 * frac:.1f}% of the retrieved text")


def main() -> None:
    from transformers import AutoModelForMultipleChoice, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        cap_memory_fraction(0.975)
    print(f"device {device}, baseline {BASELINE:.4f}, n={N} per cell")

    tok = AutoTokenizer.from_pretrained(PUBLIC_MODEL)
    model = AutoModelForMultipleChoice.from_pretrained(PUBLIC_MODEL).to(device).eval()

    t1 = pd.read_parquet(T1_CTX).sample(n=N, random_state=SEED).reset_index(drop=True)
    tr = pd.read_parquet(TRAIN_SRC2).sample(n=N, random_state=SEED).reset_index(drop=True)

    print("\nTRUNCATION BUDGET (what each format actually shows the model)")
    truncation_report(tok, t1, GATE_FMT, "gate format (1750 chars @ 512)")
    truncation_report(tok, t1, TRAIN_FMT, "training format (8000 chars @ 384)")

    print("\nTHREE CELLS")
    a = score(model, tok, t1, GATE_FMT, device, "A  eval file  + gate format (control)")
    b = score(model, tok, tr, GATE_FMT, device, "B  TRAIN file + gate format")
    c = score(model, tok, t1, TRAIN_FMT, device, "C  eval file  + TRAINING format")

    print("\nREADING")
    if abs(a - 0.7970) > 0.06:
        print(f"  !! cell A is {a:.4f}, not ~0.7970 -- the control moved, so something")
        print("     changed under the gate itself; fix that before reading B or C.")
    if b < 0.5:
        print(f"  B={b:.4f} near baseline while A={a:.4f}: the TRAINING FILE is the bug.")
        print("     Its labels or its context carry no usable signal, which is exactly")
        print("     why train_loss cannot fall below ln(5). No optimizer change fixes")
        print("     this -- rebuild the training pool and re-run this check first.")
    else:
        print(f"  B={b:.4f} vs A={a:.4f}: the training file is sound. Its labels and")
        print("     context support a reader, so the ln(5) train loss is NOT the data.")
    if c < 0.5 <= b:
        print(f"  C={c:.4f} collapsed vs A={a:.4f}: our TRAINING INPUT FORMAT is the bug.")
        print("     The reader sees the same rows; only the layout/truncation changed.")
        print("     Fix the format to the gate's before spending any more GPU quota.")
    elif b >= 0.5:
        print(f"  C={c:.4f} vs A={a:.4f}: format costs some accuracy but does not collapse.")
        print("     With data and format both cleared, the remaining suspect is the")
        print("     OPTIMIZER PATH itself -- and it can be tested with no GPU quota by")
        print("     checking that our own training step actually moves the trainable")
        print("     weights (grad norms > 0 on the unfrozen layers and the head).")
    print(f"\n  reference: our own trained reader 0.3840 [0.3649,0.4036], baseline {BASELINE:.4f}")


if __name__ == "__main__":
    main()
