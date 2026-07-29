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
