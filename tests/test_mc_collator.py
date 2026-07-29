import numpy as np
import torch
from transformers import AutoTokenizer

from llmsci.reader.mc import (
    DataCollatorForMultipleChoice,
    OPTION_COLUMNS,
    build_choice_texts,
    logits_to_ranked_labels,
)

TOKENIZER_NAME = "microsoft/deberta-v3-base"


def test_build_choice_texts_shapes_and_content():
    first, second = build_choice_texts("What is X?", ["a", "b", "c"], context="ctx")
    assert first == ["ctx", "ctx", "ctx"]
    assert second == ["What is X? a", "What is X? b", "What is X? c"]


def test_build_choice_texts_empty_context_for_closed_book():
    first, _ = build_choice_texts("What is X?", ["a", "b"], context="")
    assert first == ["", ""]


def test_collator_produces_batch_num_choices_seqlen_shape():
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    collator = DataCollatorForMultipleChoice(tokenizer)

    rows = [
        ("Question one?", ["opt A", "opt B", "opt C", "opt D", "opt E"], 0),
        ("Question two, a fair bit longer than the first one.", ["x", "y", "z", "w", "v"], 2),
    ]
    features = []
    for prompt, options, label in rows:
        first, second = build_choice_texts(prompt, options)
        encoded = tokenizer(first, second, truncation="only_first", max_length=32)
        item = dict(encoded)
        item["label"] = label
        features.append(item)

    batch = collator(features)

    batch_size, num_choices = 2, 5
    assert batch["input_ids"].shape[:2] == (batch_size, num_choices)
    assert batch["input_ids"].dim() == 3  # (batch, num_choices, seq_len)
    assert batch["attention_mask"].shape == batch["input_ids"].shape
    assert batch["labels"].shape == (batch_size,)
    assert batch["labels"].tolist() == [0, 2]
    assert batch["input_ids"].dtype == torch.int64


def test_collator_pads_to_the_longest_choice_across_the_whole_batch():
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    collator = DataCollatorForMultipleChoice(tokenizer)

    short = tokenizer(["ctx"] * 2, ["short a", "short b"], truncation="only_first", max_length=32)
    long_pair = tokenizer(
        ["ctx"] * 2,
        ["a much longer option that will tokenize to more subword pieces than the short ones", "short b"],
        truncation="only_first",
        max_length=32,
    )
    short_item, long_item = dict(short), dict(long_pair)
    short_item["label"] = 0
    long_item["label"] = 1

    batch = collator([short_item, long_item])
    seq_len = batch["input_ids"].shape[-1]
    # every choice in every example must be padded to the same seq_len
    assert batch["input_ids"].shape == (2, 2, seq_len)
    assert batch["attention_mask"].shape == (2, 2, seq_len)


def test_logits_to_ranked_labels_orders_by_descending_logit():
    logits = np.array(
        [
            [0.1, 0.9, 0.05, 0.05, 0.0],  # B > A > (C=D) > E
            [0.9, 0.0, 0.0, 0.0, 0.1],  # A > E > ...
        ]
    )
    ranked = logits_to_ranked_labels(logits, k=3)
    assert ranked[0][:2] == ["B", "A"]
    assert ranked[1][:2] == ["A", "E"]
    assert all(len(row) == 3 for row in ranked)


def test_logits_to_ranked_labels_respects_k():
    logits = np.array([[0.1, 0.2, 0.3, 0.4, 0.5]])
    assert logits_to_ranked_labels(logits, k=1) == [["E"]]
    assert logits_to_ranked_labels(logits, k=5)[0] == ["E", "D", "C", "B", "A"]
