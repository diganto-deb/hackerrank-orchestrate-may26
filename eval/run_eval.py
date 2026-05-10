#!/usr/bin/env python3
"""Run the triage pipeline on ground_truth.csv and score each row with an LLM judge."""
from __future__ import annotations

import csv
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import yaml


from corpus.index import load_master_index
from corpus.reconcile import load_unified_taxonomy
from main import load_artifacts, process_ticket, _normalize_company
from schemas import Ticket
from langfuse import get_client as _lf_client
from utils.llm import LLMClient

REPO_ROOT = Path(__file__).resolve().parent.parent
CODE_DIR = REPO_ROOT / "code"
EVAL_DIR = Path(__file__).resolve().parent
GROUND_TRUTH = EVAL_DIR / "ground_truth.csv"
JUDGE_PROMPT_PATH = EVAL_DIR / "judge_prompts" / "v1.yaml"
RESULTS_DIR = EVAL_DIR / "results"

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s", force=True)

_SCORE_CRITERIA = [
    "status_correct",
    "product_area_match",
    "response_grounded",
    "scope_respected",
    "coherence",
]


def _load_judge_template() -> str:
    with JUDGE_PROMPT_PATH.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data["template"]


def _judge_row(
    row: dict,
    output: dict,
    judge_template: str,
    llm: LLMClient,
) -> dict:
    prompt = judge_template.format(
        expected_status=row["expected_status"],
        expected_product_area=row["expected_product_area"],
        expected_request_type=row["expected_request_type"],
        status=output["status"],
        product_area=output["product_area"],
        request_type=output["request_type"],
        response=output["response"],
        justification=output["justification"],
    )
    raw = llm.call_v3([{"role": "user", "content": prompt}], json_mode=True)
    try:
        scores = json.loads(raw)
    except json.JSONDecodeError:
        scores = {c: 0.0 for c in _SCORE_CRITERIA}
        scores["aggregate"] = 0.0
        scores["notes"] = f"judge parse error: {raw[:200]}"
    scores["aggregate"] = sum(scores.get(c, 0.0) for c in _SCORE_CRITERIA) / len(_SCORE_CRITERIA)
    return scores


def main() -> None:
    if not (CODE_DIR / "corpus_cache" / "cache_manifest.json").exists():
        print("Corpus cache not found. Build it first:")
        print("  uv run python code/build_corpus.py")
        sys.exit(1)

    if not GROUND_TRUTH.exists():
        print(f"Ground truth not found at {GROUND_TRUTH}")
        sys.exit(1)

    unified, master, indexes = load_artifacts()
    llm = LLMClient()
    judge_template = _load_judge_template()

    with GROUND_TRUTH.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    results: list[dict] = []
    for idx, row in enumerate(rows, start=1):
        ticket = Ticket(
            issue=row["issue"],
            subject=row.get("subject", ""),
            company=_normalize_company(row.get("company", "") or ""),
        )
        print(f"[{idx}/{len(rows)}] {ticket.issue[:60]}", flush=True)
        logger.info("Eval ticket %d/%d", idx, len(rows))
        output_row = process_ticket(ticket, unified, master, indexes, llm)
        output = {
            "status": output_row.status.value,
            "product_area": output_row.product_area,
            "request_type": output_row.request_type.value,
            "response": output_row.response,
            "justification": output_row.justification,
        }
        scores = _judge_row(row, output, judge_template, llm)
        results.append({
            "issue": row["issue"][:80],
            "company": row.get("company", ""),
            **output,
            **{f"score_{k}": v for k, v in scores.items()},
        })

    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d-%H")
    output_path = RESULTS_DIR / f"{timestamp}.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    aggregate_scores = [r["score_aggregate"] for r in results]
    mean_score = sum(aggregate_scores) / len(aggregate_scores) if aggregate_scores else 0.0
    _lf_client().flush()
    print(f"\nEval complete: {len(results)} tickets scored")
    print(f"Mean aggregate score: {mean_score:.3f}")
    print(f"Results written to: {output_path}")


if __name__ == "__main__":
    main()
