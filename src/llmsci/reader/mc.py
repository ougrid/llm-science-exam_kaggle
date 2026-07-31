"""AutoModelForMultipleChoice plumbing: dataset, collator, and inference helpers.

Input per option, per PLAN.md: [CLS] context [SEP] prompt + " " + option [SEP].
Closed-book readers pass context="" so the same code path serves both the
Day-1 closed-book baseline and the later open-book (retrieved-context) reader.

The model reshapes (batch, num_choices, seq_len) -> (batch*num_choices, seq_len)
internally, applies one shared scalar head, and reshapes back to
(batch, num_choices) logits -- it is not a five-output head. This module's
job is only to get the inputs into that (batch, num_choices, seq_len) shape.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

OPTION_COLUMNS = ["A", "B", "C", "D", "E"]


def build_choice_texts(prompt: str, options: list[str], context: str = "") -> tuple[list[str], list[str]]:
    """Return (first_texts, second_texts) for the tokenizer's text/text_pair, one per option."""
    first = [context] * len(options)
    second = [f"{prompt} {opt}" for opt in options]
    return first, second


class MultipleChoiceDataset(Dataset):
    """Wraps a prompt/A-E/answer DataFrame, tokenizing lazily per row.

    `context_col`, if given, names a column holding retrieved context per row
    (open-book); omit it for closed-book, which passes context="".
    """

    def __init__(
        self,
        df,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 256,
        context_col: str | None = None,
        option_columns: list[str] | None = None,
    ):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.context_col = context_col
        self.option_columns = option_columns or OPTION_COLUMNS

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx]
        options = [row[c] for c in self.option_columns]
        context = row[self.context_col] if self.context_col else ""
        first, second = build_choice_texts(row["prompt"], options, context)
        encoded = self.tokenizer(
            first, second, truncation="only_first", max_length=self.max_length
        )
        item = dict(encoded)
        if "answer" in row:
            item["label"] = self.option_columns.index(row["answer"])
        return item


@dataclass
class DataCollatorForMultipleChoice:
    """Flattens (batch, num_choices) features, pads, then unflattens back.

    HF's own multiple-choice collator isn't exported in every transformers
    version, so this is written directly rather than imported.
    """

    tokenizer: PreTrainedTokenizerBase
    padding: bool | str = True
    max_length: int | None = None
    pad_to_multiple_of: int | None = None

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        has_labels = "label" in features[0]
        labels = [f.pop("label") for f in features] if has_labels else None
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])

        flattened = [
            {k: v[i] for k, v in feature.items()} for feature in features for i in range(num_choices)
        ]
        batch = self.tokenizer.pad(
            flattened,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        if has_labels:
            batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch


def logits_to_ranked_labels(
    logits: np.ndarray, k: int = 3, option_columns: list[str] | None = None
) -> list[list[str]]:
    """Top-k option letters per row, ranked by descending logit."""
    option_columns = option_columns or OPTION_COLUMNS
    order = np.argsort(-logits, axis=1)
    return [[option_columns[i] for i in row[:k]] for row in order]


def assert_trainable_dtype(model, min_bits: int = 32) -> None:
    """Fail loudly if a model's PARAMETERS are too low-precision to train.

    Guards against a regression that silently produced four failed training runs
    in this project. `transformers` 5.x changed the `from_pretrained` default to
    follow the CHECKPOINT's stored dtype; `microsoft/deberta-v3-large` ships
    fp16, so the plain call returns fp16 *parameters* -- not fp16 compute with
    fp32 master weights, which is what mixed precision means, but genuinely
    half-precision weights that AdamW then tries to update in place. Under
    transformers 4.x the identical line returned fp32.

    Why that cannot train, in one line of arithmetic: an AdamW step is ~lr in
    magnitude at step 1, so lr=2e-5 against an fp16 ULP of 2^-16 = 1.5e-5 for a
    weight near 0.03 makes every update ~1.3 ULP. Updates round to zero or snap
    by one representable increment, the model never leaves the uniform-prediction
    fixed point, and training loss pins at ln(5) = 1.6094 for a 5-option task
    while gradients, the optimizer, and the data are all perfectly healthy.

    Inference is unaffected -- fp16 forward passes are fine and faster -- so this
    is specifically a guard for training entry points.
    """
    dtypes = {p.dtype for p in model.parameters()}
    bad = {d for d in dtypes if torch.finfo(d).bits < min_bits}
    if bad:
        example = next(iter(bad))
        ulp = float(
            torch.tensor(0.03, dtype=example).nextafter(torch.tensor(1.0, dtype=example)) - 0.03
        )
        raise RuntimeError(
            f"model parameters are {sorted(str(d) for d in bad)}, which cannot be trained: "
            f"ULP near a weight of 0.03 is {ulp:.2e}, so a typical lr=2e-5 AdamW update is "
            f"~{2e-5 / ulp:.1f} ULP and rounds away. Training loss will pin at ln(5). "
            f"Load with `from_pretrained(..., dtype=torch.float32)` and use autocast/GradScaler "
            f"if you want fp16 speed with fp32 master weights."
        )


def load_mc_model_for_training(name_or_path, device=None, dtype=torch.float32):
    """`AutoModelForMultipleChoice` with an EXPLICIT dtype, verified fp32-trainable.

    Always pass dtype explicitly rather than relying on the library default: the
    default is version-dependent (see assert_trainable_dtype) and changed under
    this project mid-flight.
    """
    from transformers import AutoModelForMultipleChoice

    model = AutoModelForMultipleChoice.from_pretrained(name_or_path, dtype=dtype)
    assert_trainable_dtype(model)
    return model.to(device) if device is not None else model
