"""Dataset loading, chat-template formatting, and label masking for SFT.

The dataset is stored as JSONL with one example per line:

    {
      "category": "claims_and_damages",
      "question": "How is a freight claim valued for a damaged pallet of paperboard?",
      "answer": "Freight claims for damaged paperboard are typically valued at..."
    }

For supervised fine-tuning (SFT) we want the loss computed *only* over the
assistant's answer tokens, not the question. We do that by setting the
question tokens' labels to -100 (the ignore index in PyTorch CE loss).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict

logger = logging.getLogger(__name__)

# Chat-template role names - Qwen / Llama / Mistral all use the same OpenAI-style
# convention, so this code is portable across models.
SYSTEM_PROMPT = (
    "You are a logistics and supply-chain operations expert. Answer questions "
    "about freight, carriers, claims, routing, SOPs, and compliance precisely "
    "and concisely. If you are uncertain, say so."
)


# ---------- IO ----------


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL file into a list of dicts. Skips blank lines."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for i, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Malformed JSON at {path}:{i}: {e}") from e
    return records


def write_jsonl(records: list[dict[str, Any]], path: str | Path) -> None:
    """Write a list of dicts to JSONL."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------- Formatting ----------


def to_chat(example: dict[str, Any]) -> list[dict[str, str]]:
    """Convert one Q&A record into the chat-message format expected by chat
    templates (Qwen, Llama-3, Mistral instruct, etc.).
    """
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": example["question"]},
        {"role": "assistant", "content": example["answer"]},
    ]


def format_for_sft(example: dict[str, Any], tokenizer) -> dict[str, Any]:
    """Apply the model's chat template and return tokenized + label-masked.

    Labels on the prompt portion (system + user) are set to -100 so the loss is
    computed only over the assistant's answer.
    """
    messages = to_chat(example)

    # Render the full conversation (with assistant turn).
    full_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    # Render just the prompt (system + user, *no* assistant turn) so we can
    # measure how many tokens belong to the prompt vs. the response.
    prompt_text = tokenizer.apply_chat_template(
        messages[:-1],
        tokenize=False,
        add_generation_prompt=True,
    )

    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]

    labels = list(full_ids)
    # Mask out everything up to and including the prompt - those tokens don't
    # contribute to the loss.
    for i in range(min(len(prompt_ids), len(labels))):
        labels[i] = -100

    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
    }


# ---------- Loading ----------


def load_splits(
    train_path: str | Path,
    val_path: str | Path,
    test_path: str | Path,
) -> DatasetDict:
    """Load train/val/test JSONL files into a HuggingFace DatasetDict."""
    train = Dataset.from_list(load_jsonl(train_path))
    val = Dataset.from_list(load_jsonl(val_path))
    test = Dataset.from_list(load_jsonl(test_path))

    logger.info(
        "Loaded splits | train=%d, val=%d, test=%d",
        len(train),
        len(val),
        len(test),
    )

    # Sanity check: all examples have the expected keys.
    required = {"category", "question", "answer"}
    for name, ds in (("train", train), ("val", val), ("test", test)):
        first = ds[0]
        missing = required - set(first.keys())
        if missing:
            raise ValueError(f"{name} split missing keys: {missing}")

    return DatasetDict({"train": train, "validation": val, "test": test})


def tokenize_dataset(dataset, tokenizer, max_length: int):
    """Apply chat template + label masking + length truncation to a dataset."""

    def _process(example):
        out = format_for_sft(example, tokenizer)
        # Truncate long examples. We truncate from the *end* - for instruction
        # data with short prompts and long answers this could lose the tail of
        # the answer; that's preferable to losing the prompt.
        if len(out["input_ids"]) > max_length:
            out["input_ids"] = out["input_ids"][:max_length]
            out["attention_mask"] = out["attention_mask"][:max_length]
            out["labels"] = out["labels"][:max_length]
        return out

    return dataset.map(
        _process,
        remove_columns=dataset.column_names,
        desc="Tokenizing + masking labels",
    )
