#!/usr/bin/env python3
"""Promote reviewed rows from review_queue.csv into ground_truth.csv."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
REVIEW_QUEUE = EVAL_DIR / "review_queue.csv"
GROUND_TRUTH = EVAL_DIR / "ground_truth.csv"

_REQUIRED_FIELDS = ("expected_status", "expected_product_area", "expected_request_type")
_GROUND_TRUTH_FIELDS = [
    "issue", "subject", "company",
    "expected_status", "expected_product_area", "expected_request_type", "notes",
]
_QUEUE_FIELDS = [
    "issue", "subject", "company",
    "observed_status", "observed_product_area", "observed_request_type",
    "observed_response", "observed_justification",
    "feedback_score", "flagged",
    "expected_status", "expected_product_area", "expected_request_type", "notes",
]


def _is_complete(row: dict) -> bool:
    return all(row.get(field, "").strip() for field in _REQUIRED_FIELDS)


def main() -> None:
    if not REVIEW_QUEUE.exists() or REVIEW_QUEUE.stat().st_size == 0:
        print("review_queue.csv is empty — nothing to promote.")
        sys.exit(0)

    existing_issues: set[str] = set()
    if GROUND_TRUTH.exists():
        with GROUND_TRUTH.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                existing_issues.add(row["issue"])

    to_promote: list[dict] = []
    remaining: list[dict] = []
    with REVIEW_QUEUE.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if _is_complete(row) and row["issue"] not in existing_issues:
                to_promote.append(row)
            else:
                remaining.append(row)

    if not to_promote:
        print("No complete rows to promote (fill in expected_* fields in review_queue.csv first).")
        sys.exit(0)

    gt_exists = GROUND_TRUTH.exists() and GROUND_TRUTH.stat().st_size > 0
    with GROUND_TRUTH.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_GROUND_TRUTH_FIELDS)
        if not gt_exists:
            writer.writeheader()
        for row in to_promote:
            writer.writerow({field: row.get(field, "") for field in _GROUND_TRUTH_FIELDS})

    with REVIEW_QUEUE.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_QUEUE_FIELDS)
        writer.writeheader()
        writer.writerows(remaining)

    print(f"Promoted {len(to_promote)} rows to ground_truth.csv")
    print(f"{len(remaining)} rows remain in review_queue.csv")


if __name__ == "__main__":
    main()
