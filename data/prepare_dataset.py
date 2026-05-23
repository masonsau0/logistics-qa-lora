"""Synthetic logistics Q&A dataset generator.

Calls the Anthropic API to produce question/answer pairs across six logistics
categories defined in `data/seed_topics.json`. Output is JSONL, one example
per line:

    {
      "category": "claims_and_damages",
      "question": "...",
      "answer": "...",
      "key_facts": ["...", "..."]   # used by the evaluator
    }

Run:

    export ANTHROPIC_API_KEY=sk-ant-...
    python -m data.prepare_dataset --target 12000 --batch-size 8

Features:
  - Resumable: skips any examples already in the output file (matched by
    question hash).
  - Deduplicating: question hashes tracked across the whole run.
  - Validating: every example must parse as JSON and contain the required
    fields; malformed responses are retried.
  - Throttled: exponential backoff on rate limits.

Cost estimate (Claude Haiku, mid-2026 pricing):
  ~12K examples × ~600 tokens each ≈ 7.2M tokens ≈ $5-15 depending on model.
  Use --model claude-haiku-4-5-20251001 for cheapest. Use --model claude-sonnet-4-6
  for higher-quality answers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

from src.config import DATA

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ---------- Constants ----------

REPO_ROOT = Path(__file__).resolve().parent.parent
SEED_PATH = REPO_ROOT / "data" / "seed_topics.json"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "raw_generated.jsonl"


# ---------- Prompt builder ----------

GENERATION_INSTRUCTIONS = """\
You are generating training data for a domain Q&A model focused on logistics
and supply chain operations. Produce {batch_size} question/answer pairs in the
category: **{category}**.

Category description: {description}

Aim for variety across these subtopics:
{subtopics_list}

REQUIREMENTS for each pair:
  - Questions must be realistic — the kind an analyst, dispatcher, or ops
    manager would actually ask. Mix easy and hard. Avoid yes/no questions.
  - Answers must be factually correct, specific, and concise (60–250 words).
    Include numbers, regulations, and named methods where relevant.
  - For each answer, also list 3–6 "key_facts": short strings naming the
    specific claims that MUST appear in any correct response (used for
    evaluation). Each key_fact should be a short noun phrase or number, NOT
    a full sentence. Example: "Carmack Amendment", "9 months", "BOL".
  - Do not duplicate questions you have already generated in this batch.

Return ONLY a JSON array. No markdown, no commentary. Schema:

[
  {{
    "question": "string",
    "answer": "string",
    "key_facts": ["string", "string", ...]
  }},
  ...
]
"""


def build_prompt(category: str, spec: dict[str, Any], batch_size: int) -> str:
    subtopics = spec.get("subtopics", [])
    subtopics_list = "\n".join(f"  - {s}" for s in subtopics) or "  - (none specified)"
    return GENERATION_INSTRUCTIONS.format(
        batch_size=batch_size,
        category=category,
        description=spec.get("description", ""),
        subtopics_list=subtopics_list,
    )


# ---------- Anthropic client (lazy import) ----------


def get_client():
    """Construct the Anthropic client.

    SECURITY: The API key is read from the ANTHROPIC_API_KEY environment
    variable. Never hardcode keys in source; never commit a .env file.
    """
    try:
        from anthropic import Anthropic
    except ImportError as e:
        raise RuntimeError("anthropic SDK not installed. Run: pip install anthropic") from e

    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Copy .env.example to .env, fill in "
            "your key, then `export ANTHROPIC_API_KEY=$(grep ANTHROPIC_API_KEY .env | cut -d= -f2)`"
        )
    return Anthropic(api_key=key)


# ---------- Generation ----------


def call_model_with_retry(client, model: str, prompt: str, max_retries: int = 5) -> str:
    """Call the Anthropic API with exponential backoff on transient errors."""
    backoff = 2.0
    last_err: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}],
            )
            # Concatenate text blocks (Anthropic response is a list of content blocks).
            return "".join(b.text for b in response.content if hasattr(b, "text"))
        except Exception as e:  # noqa: BLE001 — we want to catch anything transient
            last_err = e
            msg = str(e).lower()
            transient = any(
                s in msg for s in ("rate", "timeout", "overload", "503", "429", "connection")
            )
            if attempt == max_retries or not transient:
                raise
            sleep_for = backoff + random.random()
            logger.warning("API error on attempt %d (%s); sleeping %.1fs", attempt, e, sleep_for)
            time.sleep(sleep_for)
            backoff *= 2
    raise RuntimeError(f"Exhausted retries; last error: {last_err}")


def parse_response(text: str, batch_size: int) -> list[dict[str, Any]]:
    """Parse the model's JSON response. Tolerates stray markdown fences."""
    cleaned = text.strip()
    # Strip ```json ... ``` fences if the model added them despite instructions.
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```", 2)[1]
        if cleaned.startswith("json\n") or cleaned.startswith("json "):
            cleaned = cleaned[5:]
        cleaned = cleaned.rsplit("```", 1)[0].strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Response is not valid JSON: {e}\nFirst 200 chars: {cleaned[:200]}"
        ) from e

    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array, got {type(data).__name__}")

    valid: list[dict[str, Any]] = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            logger.warning("Item %d is not a dict; skipping", i)
            continue
        q = item.get("question")
        a = item.get("answer")
        kf = item.get("key_facts", [])
        if not (isinstance(q, str) and isinstance(a, str) and isinstance(kf, list)):
            logger.warning("Item %d has wrong field types; skipping", i)
            continue
        if not q.strip() or not a.strip():
            continue
        valid.append(
            {
                "question": q.strip(),
                "answer": a.strip(),
                "key_facts": [str(x).strip() for x in kf if str(x).strip()],
            }
        )

    if len(valid) < batch_size // 2:
        raise ValueError(f"Too few valid items returned ({len(valid)} of {batch_size})")
    return valid


def hash_question(q: str) -> str:
    return hashlib.sha256(q.lower().strip().encode("utf-8")).hexdigest()[:16]


# ---------- Splitting ----------


def split_dataset(
    records: list[dict[str, Any]], val_frac: float, test_frac: float, seed: int = 42
) -> tuple[list, list, list]:
    """Stratified split by category."""
    rng = random.Random(seed)
    by_cat: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        by_cat.setdefault(r["category"], []).append(r)

    train, val, test = [], [], []
    for _category, items in by_cat.items():
        rng.shuffle(items)
        n = len(items)
        n_val = max(1, int(n * val_frac))
        n_test = max(1, int(n * test_frac))
        test.extend(items[:n_test])
        val.extend(items[n_test : n_test + n_val])
        train.extend(items[n_test + n_val :])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


# ---------- Main ----------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--target",
        type=int,
        default=DATA.target_total_examples,
        help="Total number of examples to generate",
    )
    p.add_argument("--batch-size", type=int, default=8, help="Examples per API call (per category)")
    p.add_argument("--model", default="claude-haiku-4-5-20251001", help="Anthropic model id")
    p.add_argument(
        "--raw-output",
        default=str(DEFAULT_OUTPUT),
        help="Where to accumulate generated records (resumable)",
    )
    p.add_argument("--seeds", default=str(SEED_PATH))
    p.add_argument(
        "--split", action="store_true", help="After generation, split into train/val/test JSONL"
    )
    p.add_argument(
        "--smoke", action="store_true", help="Smoke test: generate 1 batch per category and exit"
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    seeds = json.loads(Path(args.seeds).read_text())["categories"]
    categories = list(seeds.keys())

    raw_path = Path(args.raw_output)
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume: load existing question hashes.
    seen: set[str] = set()
    existing = 0
    if raw_path.exists():
        for line in raw_path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                seen.add(hash_question(rec["question"]))
                existing += 1
            except json.JSONDecodeError:
                continue
    if existing:
        logger.info("Resuming: %d existing records, skipping duplicates", existing)

    target_per_cat = args.batch_size if args.smoke else args.target // len(categories)

    client = get_client()
    f_out = raw_path.open("a", encoding="utf-8")

    total_new = 0
    try:
        for cat in categories:
            spec = seeds[cat]
            # Count what we already have for this category.
            existing_in_cat = (
                sum(
                    1
                    for line in raw_path.read_text().splitlines()
                    if line.strip() and json.loads(line).get("category") == cat
                )
                if raw_path.exists()
                else 0
            )
            need = max(0, target_per_cat - existing_in_cat)
            logger.info("[%s] have=%d, need=%d", cat, existing_in_cat, need)

            while need > 0:
                batch = min(args.batch_size, need)
                prompt = build_prompt(cat, spec, batch)
                try:
                    response_text = call_model_with_retry(client, args.model, prompt)
                    items = parse_response(response_text, batch)
                except Exception as e:  # noqa: BLE001
                    logger.error("[%s] batch failed: %s — continuing", cat, e)
                    continue

                added_this_batch = 0
                for item in items:
                    h = hash_question(item["question"])
                    if h in seen:
                        continue
                    seen.add(h)
                    record = {"category": cat, **item}
                    f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    f_out.flush()
                    added_this_batch += 1
                    total_new += 1
                    need -= 1
                    if need <= 0:
                        break
                logger.info("[%s] +%d (total new this run: %d)", cat, added_this_batch, total_new)
    finally:
        f_out.close()

    logger.info("Generation done. Wrote %d new records to %s", total_new, raw_path)

    if args.split:
        # Build splits.
        records = [json.loads(line) for line in raw_path.read_text().splitlines() if line.strip()]
        train, val, test = split_dataset(records, DATA.val_fraction, DATA.test_fraction)
        for split_name, recs, path in (
            ("train", train, DATA.train_path),
            ("val", val, DATA.val_path),
            ("test", test, DATA.test_path),
        ):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                for r in recs:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            logger.info("Wrote %s split: %d examples → %s", split_name, len(recs), path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
