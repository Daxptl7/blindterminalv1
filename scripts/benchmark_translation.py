#!/usr/bin/env python3
"""Benchmark BlindAssist translation with references and safety gates.

Manifest: one JSON object per line with:
  id, src, dest, source, references (string or list), optional human_adequate.

This script defaults to private/local execution. Pass --allow-cloud only when
the evaluation text is non-confidential and cloud transmission is intended.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules import translator  # noqa: E402


def _normalized(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def _ngrams(text: str, size: int) -> Counter:
    return Counter(text[i:i + size] for i in range(max(0, len(text) - size + 1)))


def chrf(candidate: str, reference: str, max_order: int = 6, beta: float = 2.0) -> float:
    """Dependency-free character F-score suitable for Indic scripts."""
    candidate, reference = _normalized(candidate), _normalized(reference)
    scores = []
    for order in range(1, max_order + 1):
        cand, ref = _ngrams(candidate, order), _ngrams(reference, order)
        cand_total, ref_total = sum(cand.values()), sum(ref.values())
        if not cand_total or not ref_total:
            continue
        overlap = sum((cand & ref).values())
        precision, recall = overlap / cand_total, overlap / ref_total
        denominator = beta * beta * precision + recall
        scores.append(
            0.0 if not denominator else
            (1 + beta * beta) * precision * recall / denominator
        )
    return statistics.mean(scores) if scores else 0.0


def _percent(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def load_manifest(path: Path) -> list[dict]:
    rows = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        row = json.loads(raw)
        required = {"id", "src", "dest", "source", "references"}
        missing = required - set(row)
        if missing:
            raise ValueError(f"line {line_no}: missing {', '.join(sorted(missing))}")
        references = row["references"]
        row["references"] = references if isinstance(references, list) else [references]
        rows.append(row)
    return rows


def run(rows: list[dict], allow_cloud: bool, min_chrf: float) -> list[dict]:
    results = []
    for row in rows:
        started = time.perf_counter()
        result = translator.translate_ex(
            row["source"], row["src"], row["dest"], privacy=not allow_cloud
        )
        elapsed = time.perf_counter() - started
        score = max((chrf(result.text, ref) for ref in row["references"]), default=0.0)
        validation_passed = bool(result.validation and result.validation.get("passed"))
        automatic_pass = bool(result.translated and validation_passed and score >= min_chrf)
        record = {
            "id": row["id"],
            "pair": f"{row['src']}->{row['dest']}",
            "translated": result.translated,
            "provider": result.source,
            "latency_s": round(elapsed, 4),
            "chrf": round(score, 4),
            "validation_passed": validation_passed,
            "automatic_pass": automatic_pass,
            "human_adequate": row.get("human_adequate"),
            "error": result.error,
        }
        results.append(record)
        print(json.dumps(record, ensure_ascii=False))
    return results


def summarize(results: list[dict], release_minimum: int) -> bool:
    print("\nSummary")
    by_pair = defaultdict(list)
    for row in results:
        by_pair[row["pair"]].append(row)

    release_ok = True
    for pair in sorted(by_pair):
        rows = by_pair[pair]
        automatic = sum(row["automatic_pass"] for row in rows) / len(rows)
        latencies = sorted(row["latency_s"] for row in rows)
        p95 = latencies[min(len(latencies) - 1, int(0.95 * len(latencies)))]
        human_rows = [row for row in rows if row["human_adequate"] is not None]
        human = (
            sum(bool(row["human_adequate"]) for row in human_rows) / len(human_rows)
            if human_rows else None
        )
        enough = len(rows) >= release_minimum and len(human_rows) >= release_minimum
        pair_ok = enough and automatic >= 0.95 and human is not None and human >= 0.95
        release_ok &= pair_ok
        print(
            f"  {pair}: n={len(rows)}, automatic={_percent(automatic)}, "
            f"human={_percent(human) if human is not None else 'not scored'}, "
            f"p95={p95:.3f}s, release_gate={'PASS' if pair_ok else 'NOT READY'}"
        )

    expected_pairs = {"en->hi", "en->gu", "hi->en", "hi->gu", "gu->en", "gu->hi"}
    missing = expected_pairs - set(by_pair)
    if missing:
        release_ok = False
        print("  Missing pairs: " + ", ".join(sorted(missing)))
    return release_ok


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--allow-cloud", action="store_true",
                        help="allow sending evaluation sentences to cloud providers")
    parser.add_argument("--min-chrf", type=float, default=0.50)
    parser.add_argument("--release-minimum", type=int, default=500)
    args = parser.parse_args()
    rows = load_manifest(args.manifest)
    if not rows:
        raise SystemExit("manifest contains no test cases")
    results = run(rows, args.allow_cloud, args.min_chrf)
    return 0 if summarize(results, args.release_minimum) else 2


if __name__ == "__main__":
    raise SystemExit(main())
