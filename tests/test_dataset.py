"""Unit tests for src/dataset.py.

These tests don't require a GPU or any downloaded model - they use a tiny
in-memory fake tokenizer to verify the masking and IO logic.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.dataset import (
    SYSTEM_PROMPT,
    format_for_sft,
    load_jsonl,
    to_chat,
    write_jsonl,
)


class FakeTokenizer:
    """Minimal tokenizer that emulates the apply_chat_template + __call__ API.

    Maps each character to a unique integer (a-z, space, plus a few specials).
    `apply_chat_template` returns a string with role-tagged segments so we can
    assert on the mask boundary.
    """

    PROMPT_SUFFIX = "<|assistant|>"

    def __init__(self):
        self._vocab: dict[str, int] = {}

    def _idx(self, ch: str) -> int:
        if ch not in self._vocab:
            self._vocab[ch] = len(self._vocab) + 1
        return self._vocab[ch]

    def apply_chat_template(self, messages, *, tokenize=False, add_generation_prompt=False):
        parts = []
        for m in messages:
            parts.append(f"<{m['role']}>{m['content']}</>")
        if add_generation_prompt:
            parts.append(self.PROMPT_SUFFIX)
        return "".join(parts)

    def __call__(self, text, *, add_special_tokens=False):
        return {"input_ids": [self._idx(c) for c in text]}


@pytest.fixture
def tok():
    return FakeTokenizer()


def test_to_chat_includes_system_user_assistant():
    ex = {"category": "x", "question": "Q?", "answer": "A."}
    msgs = to_chat(ex)
    assert [m["role"] for m in msgs] == ["system", "user", "assistant"]
    assert msgs[0]["content"] == SYSTEM_PROMPT
    assert msgs[1]["content"] == "Q?"
    assert msgs[2]["content"] == "A."


def test_format_for_sft_masks_prompt_tokens(tok):
    """Labels on the prompt portion must be -100, labels on the answer must equal input_ids."""
    ex = {"category": "x", "question": "Q?", "answer": "A."}
    out = format_for_sft(ex, tok)

    assert "input_ids" in out and "labels" in out and "attention_mask" in out
    assert len(out["input_ids"]) == len(out["labels"]) == len(out["attention_mask"])

    # All labels must be either -100 or equal to input_ids at the same index.
    for tok_id, label in zip(out["input_ids"], out["labels"], strict=True):
        assert label in (-100, tok_id)

    # There must be SOME prompt masked and SOME answer unmasked.
    n_masked = sum(1 for lbl in out["labels"] if lbl == -100)
    n_kept = sum(1 for lbl in out["labels"] if lbl != -100)
    assert n_masked > 0, "no prompt tokens were masked"
    assert n_kept > 0, "all tokens were masked - no answer remained"

    # The masked prefix should appear first and the kept suffix last.
    first_kept = next(i for i, lbl in enumerate(out["labels"]) if lbl != -100)
    assert all(lbl == -100 for lbl in out["labels"][:first_kept])


def test_jsonl_roundtrip(tmp_path: Path):
    records = [
        {
            "category": "claims_and_damages",
            "question": "Q1?",
            "answer": "A1.",
            "key_facts": ["a", "b"],
        },
        {"category": "freight_calculations", "question": "Q2?", "answer": "A2.", "key_facts": []},
    ]
    p = tmp_path / "test.jsonl"
    write_jsonl(records, p)
    loaded = load_jsonl(p)
    assert loaded == records


def test_jsonl_skips_blank_lines(tmp_path: Path):
    p = tmp_path / "test.jsonl"
    p.write_text(
        '{"category":"c","question":"Q","answer":"A"}\n\n   \n{"category":"c","question":"Q2","answer":"A2"}\n'
    )
    loaded = load_jsonl(p)
    assert len(loaded) == 2


def test_jsonl_raises_on_malformed_line(tmp_path: Path):
    p = tmp_path / "bad.jsonl"
    p.write_text('{"good": true}\nnot valid json\n')
    with pytest.raises(ValueError, match="Malformed JSON"):
        load_jsonl(p)


def test_jsonl_raises_when_missing(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_jsonl(tmp_path / "nonexistent.jsonl")
