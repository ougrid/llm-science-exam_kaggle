"""Reader plumbing for the cdeotte open-book reproduction.

Reimplemented from the prose in ../NOTES.md, not copied from the notebook.
Deliberately kept separate from `src/llmsci/reader/mc.py`: the main pipeline
builds `[CLS] context [SEP] prompt + " " + option [SEP]`, whereas cdeotte
builds `[CLS] context #### prompt [SEP] option [SEP]` -- the segment boundary
falls between context and "####", and there is an extra internal [SEP]
between prompt and option. Reusing the main pipeline's collator would have
silently changed the thing under reproduction.

`llmsci.metrics` IS reused (paired bootstrap, MAP@3) -- CLAUDE.md requires a
single source of truth for scoring, so scores from this track stay
comparable to the main pipeline's.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoModelForMultipleChoice, PreTrainedTokenizerBase

OPTIONS = ["A", "B", "C", "D", "E"]

# cdeotte part 1, verbatim config values (see ../NOTES.md "Training recipe")
MODEL_NAME = "microsoft/deberta-v3-large"
MAX_INPUT = 256
FREEZE_LAYERS = 18
FREEZE_EMBEDDINGS = True
NUM_TRAIN_SAMPLES = 1_024
EPOCHS = 2
LR = 2e-5
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
# cdeotte: per_device_train_batch_size=1, grad_accum=8, on 2 GPUs -> effective 16.
# One GPU here, so accumulate 16 to match the effective batch rather than the
# literal accumulation count.
MICRO_BATCH = 1
GRAD_ACCUM_STEPS = 16
# HF Trainer's default adam_epsilon is 1e-8, which is what cdeotte inherits.
# That default makes the FIRST optimizer step turn all 103 trainable tensors
# non-finite on this box (torch 2.11 + sm_120), with a perfectly healthy loss
# (1.54) and pre-clip grad norm (1.55) going in -- see debug_nan.py /
# debug_optim.py, and DEVLOG.md's independent root-cause of the same failure as
# Adam's first-step instability (near-zero bias-corrected second moment).
# eps=1e-6 is Microsoft's own documented DeBERTa recipe and the value the main
# pipeline settled on, so using it keeps this track's optimizer identical to
# the main pipeline's and leaves the recipe as the only difference between them.
# Measured alternative: AdamW(fused=True) also survives at eps=1e-8.
ADAM_EPS = 1e-6


def build_choice_texts(prompt: str, options: list[str], context: str) -> tuple[list[str], list[str]]:
    """cdeotte's per-option text pair.

    first  = "[CLS] " + context
    second = " #### " + prompt + " [SEP] " + option + " [SEP]"

    Tokenized with add_special_tokens=False, so the literal "[CLS]"/"[SEP]"
    strings are what produce the special token ids. This reproduces the layout
    add_special_tokens=True would give while leaving truncation='only_first'
    free to cut the context and never the question or the option.
    """
    first = [f"[CLS] {context}"] * len(options)
    second = [f" #### {prompt} [SEP] {opt} [SEP]" for opt in options]
    return first, second


class OpenBookMCDataset(Dataset):
    """Tokenizes prompt/A-E/context rows lazily into cdeotte's MC format."""

    def __init__(
        self,
        df,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = MAX_INPUT,
        context_col: str = "context",
    ):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.context_col = context_col

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx]
        options = [str(row[c]) for c in OPTIONS]
        first, second = build_choice_texts(str(row["prompt"]), options, str(row[self.context_col]))
        encoded = self.tokenizer(
            first,
            second,
            truncation="only_first",
            max_length=self.max_length,
            add_special_tokens=False,
        )
        item = dict(encoded)
        if "answer" in row and row["answer"] in OPTIONS:
            item["label"] = OPTIONS.index(row["answer"])
        return item


@dataclass
class MCCollator:
    """Flatten (batch, 5) features, pad, unflatten to (batch, 5, L)."""

    tokenizer: PreTrainedTokenizerBase

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        has_labels = "label" in features[0]
        labels = [f.pop("label") for f in features] if has_labels else None
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])
        flattened = [
            {k: v[i] for k, v in feat.items()} for feat in features for i in range(num_choices)
        ]
        batch = self.tokenizer.pad(flattened, padding=True, return_tensors="pt")
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        if has_labels:
            batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch


def logits_to_ranked_labels(logits: np.ndarray, k: int = 3) -> list[list[str]]:
    order = np.argsort(-logits, axis=1)
    return [[OPTIONS[i] for i in row[:k]] for row in order]


def build_model(device: torch.device, freeze_layers: int = FREEZE_LAYERS,
                freeze_embeddings: bool = FREEZE_EMBEDDINGS):
    """Load deberta-v3-large for MC and apply cdeotte's freezing.

    The freezing is a memory measure, not a quality one (cdeotte lists it
    under "tricks to train models efficiently" and notes accuracy may drop).
    It is load-bearing here: with 24/24 layers and the 128100x1024 embedding
    matrix trainable, AdamW state alone is ~3.5 GB and the run will not fit
    in 8.55 GB.
    """
    model = AutoModelForMultipleChoice.from_pretrained(MODEL_NAME).to(device)
    backbone = getattr(model, "deberta", None)
    if backbone is None:
        raise RuntimeError(
            "expected a `.deberta` submodule on AutoModelForMultipleChoice; the "
            "transformers 5.x layout differs from the 4.31 the notebook assumed -- "
            "inspect model.named_children() before changing the freezing logic"
        )
    if freeze_embeddings:
        for p in backbone.embeddings.parameters():
            p.requires_grad = False
    if freeze_layers > 0:
        layers = backbone.encoder.layer
        if freeze_layers > len(layers):
            raise ValueError(f"asked to freeze {freeze_layers} of {len(layers)} layers")
        for layer in layers[:freeze_layers]:
            for p in layer.parameters():
                p.requires_grad = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[model] trainable {trainable/1e6:.1f}M / {total/1e6:.1f}M params "
          f"(froze embeddings={freeze_embeddings}, first {freeze_layers} layers)")
    return model


@torch.no_grad()
def predict_logits(model, tokenizer, df, device, batch_size: int = 4,
                   max_length: int = MAX_INPUT, context_col: str = "context",
                   label: str = "eval") -> np.ndarray:
    """Batched inference -> (n, 5) logits, with periodic progress.

    Progress printing is not cosmetic: CLAUDE.md requires long GPU loops to
    print elapsed/rate/ETA, because a silent multi-minute gap is
    indistinguishable from a hang under WSL2.
    """
    from torch.utils.data import DataLoader

    was_training = model.training
    model.eval()
    ds = OpenBookMCDataset(df, tokenizer, max_length=max_length, context_col=context_col)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=MCCollator(tokenizer))
    out, start = [], time.time()
    n_batches = len(loader)
    for i, batch in enumerate(loader):
        batch.pop("labels", None)
        batch = {k: v.to(device) for k, v in batch.items()}
        out.append(model(**batch).logits.float().cpu().numpy())
        if (i + 1) % 100 == 0 or (i + 1) == n_batches:
            done = time.time() - start
            rate = (i + 1) / done
            eta = (n_batches - i - 1) / rate
            print(f"  [{label}] batch {i+1}/{n_batches} elapsed {done:.0f}s "
                  f"rate {rate:.1f} b/s eta {eta:.0f}s", flush=True)
    if was_training:
        model.train()
    return np.concatenate(out, axis=0)
