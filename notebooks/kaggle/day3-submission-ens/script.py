"""Offline submission v2: our retrieval + an ENSEMBLE of PUBLIC readers,
with the config chosen by self-measurement rather than by hope.

PROVENANCE -- READ BEFORE QUOTING ANY SCORE.
  * OURS: the corpus (jjinho/wikipedia-20230701 sampled to 1.6M chunks), the
    chunking + title-prefixing, the BM25 index, query construction, this
    offline pipeline, and the model-selection protocol below.
  * NOT OURS: both readers. `mgoksu/llm-science-run-context-2` and
    `sandiago21/llm-science-exam-deberta-v3-large-context-3` are public
    checkpoints fine-tuned by other competitors. Public models/datasets were
    explicitly permitted and were standard practice in this competition, so
    this is a legitimate submission -- but it is NOT evidence of a model we
    trained. Cited in CREDITS.md.

WHY AN ENSEMBLE: cdeotte part 2 -- the notebook behind the published 0.823761
-- was itself a 50/50 two-leg ensemble, so averaging independently-trained
readers is the known-good move here rather than a guess.

WHY IT SELF-MEASURES: `train.csv` in the competition data is the official 200
rows WITH answers, and it is clean with respect to these checkpoints' public
training pools (zero prompt overlap, asserted in
scripts/build_context_train_pool.py). So the kernel scores every candidate
config on that clean gold 200, prints each with a bootstrap CI, and writes the
submission from whichever config actually wins -- no config is shipped on
faith. Single-model reference to beat: MAP@3 0.8592 [0.8200, 0.8958].

Degrades gracefully at every step: a baseline submission is written before any
heavy work, a failed second checkpoint falls back to the single-model config,
and an unusable GPU falls back to CPU.
"""

from __future__ import annotations

import gc
import os
import subprocess
import sys
import time
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

START = time.time()
TIME_BUDGET_S = 7.5 * 3600  # competition limit is 9 h; leave headroom
OUT = Path("/kaggle/working")
OPTIONS = ["A", "B", "C", "D", "E"]
TOP_K = 5
CONTEXT_CHAR_CLIP = 1750  # matches the readers' own part-2 inference recipe
MODEL_MARKERS = [
    ("mgoksu", "llm-science-run-context-2"),
    ("sandiago21", "llm-science-exam-deberta-v3-large-context-3"),
]
BATCH_SIZE = 2  # v2 OOM'd on a T4 at 8; 200-4,000 rows is fast either way


def find_path(filename: str) -> Path:
    matches = glob(f"/kaggle/input/**/{filename}", recursive=True)
    if not matches:
        print("DEBUG /kaggle/input tree:")
        for p in sorted(glob("/kaggle/input/**/*", recursive=True))[:200]:
            print(" ", p)
        raise FileNotFoundError(filename)
    return Path(matches[0])


def find_dir(marker: str) -> Path:
    """Directory containing `marker` (e.g. a config.json for the model)."""
    return find_path(marker).parent


def install_bm25s() -> None:
    wheel = glob("/kaggle/input/**/bm25s-*.whl", recursive=True)
    if not wheel:
        raise FileNotFoundError("bm25s wheel not found in attached datasets")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", "-q", wheel[0]]
    )
    print(f"installed {os.path.basename(wheel[0])} offline")


def write_submission(df: pd.DataFrame, preds: list[str]) -> None:
    sub = pd.DataFrame({"id": df["id"].values, "prediction": preds})
    assert len(sub) == len(df), (len(sub), len(df))
    assert sub["prediction"].map(lambda s: 1 <= len(s.split()) <= 3).all()
    assert sub["prediction"].map(lambda s: all(t in OPTIONS for t in s.split())).all()
    assert sub["prediction"].map(lambda s: len(set(s.split())) == len(s.split())).all()
    sub.to_csv(OUT / "submission.csv", index=False)
    print(f"wrote submission.csv with {len(sub)} rows")
    print(sub.head(3).to_string(index=False))


def main() -> None:
    install_bm25s()
    import bm25s  # noqa: E402  (installed above)
    import torch  # noqa: E402
    from torch.utils.data import DataLoader, Dataset  # noqa: E402
    from transformers import AutoModelForMultipleChoice, AutoTokenizer  # noqa: E402

    test = pd.read_csv(find_path("test.csv"))
    print(f"test rows: {len(test)}  ({'COMMIT phase' if len(test) <= 400 else 'SUBMIT phase'})")

    # Fallback written immediately: if anything below fails or runs long, a
    # valid full-length submission already exists on disk. "A B C" scores the
    # analytic random baseline (0.3667) rather than erroring out.
    write_submission(test, ["A B C"] * len(test))

    index_dir = find_dir("params.index.json")
    print(f"BM25 index dir: {index_dir}")
    chunks = pd.read_parquet(index_dir / "chunk_texts.parquet", columns=["text"])
    chunk_texts = chunks["text"].tolist()
    del chunks
    retriever = bm25s.BM25.load(str(index_dir))
    print(f"loaded index over {len(chunk_texts)} chunks [{time.time() - START:.0f}s]")

    # Identical to src/llmsci/retrieve/sparse.py build_query(), so this
    # retrieves exactly what the local 0.8592 gold-200 measurement retrieved.
    def attach_context(df: pd.DataFrame) -> list[str]:
        qs = [f"{r['prompt']} " + " ".join(str(r[c]) for c in OPTIONS) for _, r in df.iterrows()]
        tk = bm25s.tokenize(qs, stopwords="en", show_progress=False)
        ix, _ = retriever.retrieve(tk, k=TOP_K, show_progress=False)
        print(f"  retrieved top-{TOP_K} for {len(df)} rows [{time.time() - START:.0f}s]", flush=True)
        return [" ".join(chunk_texts[j] for j in row) for row in ix]

    test = test.copy()
    test["context"] = attach_context(test)

    device = torch.device("cpu")
    if torch.cuda.is_available():
        try:
            _p = torch.zeros(8, 8, device="cuda")
            _ = (_p @ _p).sum().item()
            torch.cuda.synchronize()
            device = torch.device("cuda")
            print(f"GPU usable: {torch.cuda.get_device_name(0)} cc={torch.cuda.get_device_capability(0)}")
        except Exception as e:
            print(f"GPU present but UNUSABLE ({type(e).__name__}: {e}) -- CPU fallback")
    print(f"device: {device}")

    # Gold 200 (official train.csv) carries answers and is clean w.r.t. these
    # public checkpoints -- our measurement surface for picking a config.
    gold = pd.read_csv(find_path("train.csv"))
    gold["context"] = attach_context(gold)
    print(f"gold rows: {len(gold)} (clean, with answers)")

    def logits_for(model_dir: Path, df: pd.DataFrame, tokenizer) -> np.ndarray:
        class DS(Dataset):
            def __init__(self, d):
                self.d = d.reset_index(drop=True)
            def __len__(self):
                return len(self.d)
            def __getitem__(self, i):
                r = self.d.iloc[i]
                first = f"{str(r['context'])[:CONTEXT_CHAR_CLIP]} #### {r['prompt']}"
                return dict(tokenizer([first] * 5, [str(r[c]) for c in OPTIONS],
                                      truncation="only_first", max_length=512))

        def collate(feats):
            n = len(feats)
            flat = [{k: v[i] for k, v in f.items()} for f in feats for i in range(5)]
            b = tokenizer.pad(flat, padding=True, return_tensors="pt")
            return {k: v.view(n, 5, -1) for k, v in b.items()}

        model = AutoModelForMultipleChoice.from_pretrained(model_dir).to(device).eval()
        loader = DataLoader(DS(df), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)
        out = []
        with torch.no_grad():
            for i, batch in enumerate(loader):
                try:
                    b = {k: v.to(device) for k, v in batch.items()}
                    out.append(model(**b).logits.float().cpu().numpy())
                except torch.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    out.append(model.to("cpu")(**batch).logits.float().numpy())
                    model.to(device)
                if (i + 1) % 50 == 0 or (i + 1) == len(loader):
                    print(f"    batch {i+1}/{len(loader)} [{time.time()-START:.0f}s]", flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        return np.concatenate(out, axis=0)

    def map3(df: pd.DataFrame, logits: np.ndarray):
        order = np.argsort(-logits, axis=1)[:, :3]
        preds = [[OPTIONS[j] for j in row] for row in order]
        sc = []
        for (_, r), p in zip(df.iterrows(), preds):
            sc.append(next((1.0 / (i + 1) for i, o in enumerate(p) if o == r["answer"]), 0.0))
        sc = np.array(sc)
        rng = np.random.default_rng(0)
        boots = np.array([sc[rng.integers(0, len(sc), len(sc))].mean() for _ in range(10_000)])
        return sc.mean(), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))

    gold_logits, test_logits, used = [], [], []
    for owner, name in MODEL_MARKERS:
        try:
            cands = glob(f"/kaggle/input/**/config.json", recursive=True)
            md = next((Path(c).parent for c in cands if owner in c or name in c), None)
            if md is None:
                print(f"  SKIP {owner}/{name}: not attached")
                continue
            print(f"  reader: {md} [{time.time()-START:.0f}s]", flush=True)
            tok = AutoTokenizer.from_pretrained(md)
            gl = logits_for(md, gold, tok)
            tl = logits_for(md, test, tok)
            m, lo, hi = map3(gold, gl)
            print(f"  >> {name} gold-200 MAP@3 {m:.4f} [{lo:.4f},{hi:.4f}]", flush=True)
            gold_logits.append(gl); test_logits.append(tl); used.append(name)
        except Exception as e:
            print(f"  FAILED {owner}/{name}: {type(e).__name__}: {e} -- continuing", flush=True)

    if not gold_logits:
        print("no reader usable -- keeping the baseline fallback submission")
        return

    # Candidate configs, all scored on the SAME clean gold 200.
    configs = {}
    for nm, gl, tl in zip(used, gold_logits, test_logits):
        configs[f"single:{nm}"] = (gl, tl)
    if len(gold_logits) > 1:
        def z(a):  # z-normalise per row before averaging: the two readers'
            m = a.mean(axis=1, keepdims=True)  # logit scales are not comparable
            s = a.std(axis=1, keepdims=True) + 1e-9
            return (a - m) / s
        configs["ensemble:mean-raw"] = (
            np.mean(gold_logits, axis=0), np.mean(test_logits, axis=0))
        configs["ensemble:mean-znorm"] = (
            np.mean([z(a) for a in gold_logits], axis=0),
            np.mean([z(a) for a in test_logits], axis=0))

    print("\n=== gold-200 comparison (single-model reference: 0.8592 [0.8200,0.8958]) ===")
    best_name, best_score, best_test = None, -1.0, None
    for nm, (gl, tl) in configs.items():
        m, lo, hi = map3(gold, gl)
        print(f"  {nm:28s} MAP@3 {m:.4f} [{lo:.4f},{hi:.4f}]")
        if m > best_score:
            best_name, best_score, best_test = nm, m, tl

    print(f"\nSELECTED: {best_name} (gold-200 MAP@3 {best_score:.4f})")
    order = np.argsort(-best_test, axis=1)[:, :3]
    write_submission(test, [" ".join(OPTIONS[j] for j in row) for row in order])
    print(f"done [{time.time() - START:.0f}s]")


if __name__ == "__main__":
    main()
