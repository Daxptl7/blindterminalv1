#!/usr/bin/env python3
"""Measure BlindAssist OCR accuracy against transcribed textbook pages."""

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import statistics
import sys
import time

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules import ocr  # noqa: E402


def normalize_text(text):
    return re.sub(r"\s+", " ", str(text or "")).strip().casefold()


def edit_distance(left, right):
    """Memory-bounded Levenshtein distance."""
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for row, left_char in enumerate(left, start=1):
        current = [row]
        for column, right_char in enumerate(right, start=1):
            current.append(min(
                current[-1] + 1,
                previous[column] + 1,
                previous[column - 1] + (left_char != right_char),
            ))
        previous = current
    return previous[-1]


def matched_count(expected_tokens, actual_tokens):
    expected = Counter(expected_tokens)
    actual = Counter(actual_tokens)
    return sum((expected & actual).values())


def _percent(value):
    return f"{value * 100:.2f}%"


def _read_ground_truth(entry, base_dir):
    if "text" in entry:
        return str(entry["text"])
    ground_truth = base_dir / entry["ground_truth"]
    return ground_truth.read_text(encoding="utf-8")


def _recognize(image, engine_name, lang):
    if engine_name == "tesseract":
        return ocr._tesseract_engine.extract_text(image, lang=lang)
    if engine_name == "gemini":
        return ocr._gemini_engine.extract_text(image, lang=lang)
    return ocr._engine.extract_text(image, lang=lang)


def evaluate(manifest_path, engine_name):
    manifest_path = Path(manifest_path).resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    pages = payload.get("pages", []) if isinstance(payload, dict) else payload
    if not pages:
        raise ValueError("Manifest must contain at least one page")

    totals = {
        "characters": 0,
        "character_errors": 0,
        "words": 0,
        "matched_words": 0,
        "numbers": 0,
        "matched_numbers": 0,
    }
    latencies = []
    results = []

    for index, entry in enumerate(pages, start=1):
        image_path = manifest_path.parent / entry["image"]
        expected_raw = _read_ground_truth(entry, manifest_path.parent)
        lang = entry.get("lang", "eng")

        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
        started = time.perf_counter()
        actual_raw = _recognize(image, engine_name, lang)
        latency = time.perf_counter() - started

        expected = normalize_text(expected_raw)
        actual = normalize_text(actual_raw)
        expected_words = re.findall(r"\w+", expected, flags=re.UNICODE)
        actual_words = re.findall(r"\w+", actual, flags=re.UNICODE)
        expected_numbers = re.findall(r"\d+(?:[.,]\d+)*", expected)
        actual_numbers = re.findall(r"\d+(?:[.,]\d+)*", actual)
        errors = edit_distance(expected, actual)
        words_matched = matched_count(expected_words, actual_words)
        numbers_matched = matched_count(expected_numbers, actual_numbers)

        totals["characters"] += len(expected)
        totals["character_errors"] += errors
        totals["words"] += len(expected_words)
        totals["matched_words"] += words_matched
        totals["numbers"] += len(expected_numbers)
        totals["matched_numbers"] += numbers_matched
        latencies.append(latency)

        page_cer = errors / max(len(expected), 1)
        page_recall = words_matched / max(len(expected_words), 1)
        results.append((str(image_path), page_cer, page_recall, latency))
        print(
            f"[{index}/{len(pages)}] {image_path.name}: "
            f"CER={_percent(page_cer)}, word recall={_percent(page_recall)}, "
            f"latency={latency:.2f}s"
        )

    sorted_latencies = sorted(latencies)
    p95_index = max(0, int(len(sorted_latencies) * 0.95 + 0.9999) - 1)
    summary = {
        "pages": len(pages),
        "character_error_rate": totals["character_errors"] / max(totals["characters"], 1),
        "word_recall": totals["matched_words"] / max(totals["words"], 1),
        "numeric_accuracy": (
            totals["matched_numbers"] / totals["numbers"]
            if totals["numbers"]
            else None
        ),
        "mean_latency_seconds": statistics.mean(latencies),
        "p95_latency_seconds": sorted_latencies[p95_index],
        "results": results,
    }
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark OCR against page images with human ground truth."
    )
    parser.add_argument("manifest", help="Path to the benchmark manifest JSON")
    parser.add_argument(
        "--engine",
        choices=("tesseract", "gemini", "surya"),
        default="tesseract",
    )
    parser.add_argument("--max-cer", type=float, default=0.05)
    parser.add_argument("--min-word-recall", type=float, default=0.95)
    parser.add_argument("--min-numeric-accuracy", type=float, default=0.99)
    args = parser.parse_args()

    summary = evaluate(args.manifest, args.engine)
    numeric = summary["numeric_accuracy"]
    print("\nAggregate result")
    print(f"Pages: {summary['pages']}")
    print(f"Character error rate: {_percent(summary['character_error_rate'])}")
    print(f"Word recall: {_percent(summary['word_recall'])}")
    print(f"Numeric accuracy: {_percent(numeric) if numeric is not None else 'N/A'}")
    print(f"Mean latency: {summary['mean_latency_seconds']:.2f}s")
    print(f"p95 latency: {summary['p95_latency_seconds']:.2f}s")

    passed = (
        summary["character_error_rate"] <= args.max_cer
        and summary["word_recall"] >= args.min_word_recall
        and (numeric is None or numeric >= args.min_numeric_accuracy)
    )
    print("Release gate: " + ("PASS" if passed else "FAIL"))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
