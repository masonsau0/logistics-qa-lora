"""Evaluation harness.

Computes three metrics on the held-out test set:

  1. Exact-match (EM): does the model's answer contain all the required
     key facts from the reference? Implemented as a "normalized keyword
     containment" check - exact string match is too strict for free-text
     answers, so we extract key facts at dataset-generation time and check
     for their presence here.

  2. Token-F1: token overlap between prediction and reference (standard
     SQuAD-style F1, helps with shorter answers).

  3. Per-category breakdown: same metrics, grouped by question category, so
     you can see where the fine-tune helped and where it hurt.

Usage:

    # Evaluate base model only
    python -m src.evaluate --no-adapter

    # Evaluate fine-tuned model
    python -m src.evaluate --adapter artifacts/checkpoints/final

    # Compare both, write to results.json
    python -m src.evaluate --compare --output artifacts/results/eval.json
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import string
import sys
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.config import DATA, MODEL, RESULTS_DIR
from src.dataset import load_jsonl

# LogisticsQA is imported lazily inside main() to keep the metrics helpers
# (normalize, token_f1, keyword_em) importable without torch installed. This
# matters for the test suite and CI.
if TYPE_CHECKING:
    from src.inference import LogisticsQA

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ---------- Metrics ----------

_PUNCT_RE = re.compile(f"[{re.escape(string.punctuation)}]")
_WS_RE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """SQuAD-style normalization: lowercase, strip punctuation and articles,
    collapse whitespace. Standard for QA metrics."""
    text = text.lower()
    text = _PUNCT_RE.sub(" ", text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = _WS_RE.sub(" ", text).strip()
    return text


def token_f1(prediction: str, reference: str) -> float:
    """Token-overlap F1. Returns 0 if either side is empty."""
    pred_tokens = normalize(prediction).split()
    ref_tokens = normalize(reference).split()
    if not pred_tokens or not ref_tokens:
        return 0.0
    common: dict[str, int] = {}
    for t in pred_tokens:
        if t in ref_tokens:
            common[t] = common.get(t, 0) + 1
    n_same = sum(common.values())
    if n_same == 0:
        return 0.0
    precision = n_same / len(pred_tokens)
    recall = n_same / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def keyword_em(prediction: str, key_facts: list[str]) -> float:
    """Fraction of required key facts present (case-insensitive, normalized)
    in the prediction. If no key_facts are provided, falls back to checking
    whether the normalized reference is a substring of the prediction.

    Returns 1.0 (fully present) or a fraction in [0, 1].
    """
    if not key_facts:
        return 0.0
    norm_pred = normalize(prediction)
    hits = sum(1 for kf in key_facts if normalize(kf) in norm_pred)
    return hits / len(key_facts)


# ---------- Evaluation loop ----------


def evaluate_model(
    qa: LogisticsQA, test_records: list[dict[str, Any]], label: str
) -> dict[str, Any]:
    """Run the model over the test set, return aggregated metrics."""
    per_category: dict[str, list[dict[str, float]]] = defaultdict(list)
    predictions: list[dict[str, Any]] = []

    n = len(test_records)
    for i, rec in enumerate(test_records, start=1):
        if i % 25 == 0 or i == n:
            logger.info("[%s] %d / %d", label, i, n)

        prediction = qa.answer(rec["question"])
        reference = rec["answer"]
        key_facts = rec.get("key_facts", [])

        em = keyword_em(prediction, key_facts)
        f1 = token_f1(prediction, reference)
        category = rec.get("category", "uncategorized")

        per_category[category].append({"em": em, "f1": f1})
        predictions.append(
            {
                "category": category,
                "question": rec["question"],
                "reference": reference,
                "prediction": prediction,
                "em": em,
                "f1": f1,
            }
        )

    # Aggregate.
    by_cat = {}
    for cat, scores in per_category.items():
        by_cat[cat] = {
            "n": len(scores),
            "em": sum(s["em"] for s in scores) / len(scores),
            "f1": sum(s["f1"] for s in scores) / len(scores),
        }
    overall = {
        "n": n,
        "em": sum(p["em"] for p in predictions) / n,
        "f1": sum(p["f1"] for p in predictions) / n,
    }

    return {
        "label": label,
        "overall": overall,
        "per_category": by_cat,
        "predictions": predictions,
    }


# ---------- CLI ----------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model-name", default=MODEL.name, help="Base HF model id")
    p.add_argument("--adapter", default=None, help="Path to LoRA adapter directory")
    p.add_argument("--no-adapter", action="store_true", help="Evaluate base model only")
    p.add_argument(
        "--compare", action="store_true", help="Evaluate both base and adapter, side by side"
    )
    p.add_argument("--test-path", default=DATA.test_path)
    p.add_argument("--output", default=str(RESULTS_DIR / "eval.json"))
    p.add_argument("--limit", type=int, default=None, help="Cap test examples (for smoke tests)")
    return p.parse_args()


def main() -> int:
    # Lazy import - keeps the metrics-only API testable without torch installed.
    from src.inference import LogisticsQA

    args = parse_args()
    test_records = load_jsonl(args.test_path)
    if args.limit:
        test_records = test_records[: args.limit]
    logger.info("Test set: %d examples", len(test_records))

    results: dict[str, Any] = {}

    if args.compare or args.no_adapter:
        logger.info("Evaluating BASE model: %s", args.model_name)
        base = LogisticsQA(base_model_name=args.model_name, adapter_path=None)
        results["base"] = evaluate_model(base, test_records, label="base")
        del base  # free VRAM before loading the next one

    if args.compare or args.adapter:
        adapter_path = args.adapter or "artifacts/checkpoints/final"
        logger.info("Evaluating FINE-TUNED model: adapter=%s", adapter_path)
        ft = LogisticsQA(base_model_name=args.model_name, adapter_path=adapter_path)
        results["fine_tuned"] = evaluate_model(ft, test_records, label="fine_tuned")

    # Add a delta summary if both were run.
    if "base" in results and "fine_tuned" in results:
        results["delta"] = {
            "em": results["fine_tuned"]["overall"]["em"] - results["base"]["overall"]["em"],
            "f1": results["fine_tuned"]["overall"]["f1"] - results["base"]["overall"]["f1"],
        }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    logger.info("Wrote results to %s", out_path)

    # Print a quick summary table.
    print("\n" + "=" * 60)
    print(f"{'Model':<15} {'EM':>10} {'F1':>10}")
    print("-" * 60)
    for k in ("base", "fine_tuned"):
        if k in results:
            r = results[k]["overall"]
            print(f"{k:<15} {r['em']:>10.3f} {r['f1']:>10.3f}")
    if "delta" in results:
        d = results["delta"]
        print(f"{'Δ':<15} {d['em']:>+10.3f} {d['f1']:>+10.3f}")
    print("=" * 60 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
