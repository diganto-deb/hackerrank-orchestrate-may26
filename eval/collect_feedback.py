#!/usr/bin/env python3
"""Collect low-scoring or flagged rows from output.csv into eval/review_queue.csv."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_CSV = REPO_ROOT / "support_tickets" / "output.csv"
REVIEW_QUEUE = Path(__file__).resolve().parent / "review_queue.csv"
SCORE_THRESHOLD = 0.5

_QUEUE_FIELDS = [
    "issue", "subject", "company",
    "observed_status", "observed_product_area", "observed_request_type",
    "observed_response", "observed_justification",
    "feedback_score", "flagged",
    "expected_status", "expected_product_area", "expected_request_type", "notes",
]


def main() -> None:
    if not OUTPUT_CSV.exists():
        print(f"output.csv not found at {OUTPUT_CSV}")
        sys.exit(1)

    existing_issues: set[str] = set()
    if REVIEW_QUEUE.exists() and REVIEW_QUEUE.stat().st_size > 0:
        with REVIEW_QUEUE.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                existing_issues.add(row["issue"])

    queue_is_empty = not REVIEW_QUEUE.exists() or REVIEW_QUEUE.stat().st_size == 0
    collected = 0

    with (
        OUTPUT_CSV.open(newline="", encoding="utf-8") as src,
        REVIEW_QUEUE.open("a", newline="", encoding="utf-8") as dst,
    ):
        reader = csv.DictReader(src)
        writer = csv.DictWriter(dst, fieldnames=_QUEUE_FIELDS)
        if queue_is_empty:
            writer.writeheader()

        for row in reader:
            issue = row.get("issue", "")
            if issue in existing_issues:
                continue
            try:
                score = float(row.get("feedback_score", "") or "1.0")
            except ValueError:
                score = 1.0
            flagged = row.get("flagged", "").strip().lower() in ("true", "1", "yes")
            if score < SCORE_THRESHOLD or flagged:
                writer.writerow({
                    "issue": issue,
                    "subject": row.get("subject", ""),
                    "company": row.get("company", ""),
                    "observed_status": row.get("status", ""),
                    "observed_product_area": row.get("product_area", ""),
                    "observed_request_type": row.get("request_type", ""),
                    "observed_response": row.get("response", ""),
                    "observed_justification": row.get("justification", ""),
                    "feedback_score": row.get("feedback_score", ""),
                    "flagged": row.get("flagged", ""),
                    "expected_status": "",
                    "expected_product_area": "",
                    "expected_request_type": "",
                    "notes": "",
                })
                collected += 1

    print(f"Collected {collected} rows into {REVIEW_QUEUE}")


if __name__ == "__main__":
    main()
