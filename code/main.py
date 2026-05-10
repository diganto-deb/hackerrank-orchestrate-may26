#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

from langfuse import get_client as _lf_client, observe as _lf_observe

from exceptions import PipelineError
from corpus.index import load_master_index
from corpus.reconcile import load_unified_taxonomy
from schemas import MasterIndexEntry, OutputRow, RequestType, Status, Ticket, UnifiedTaxonomy
from utils.prompt_registry import PromptRegistry
from utils.tracer import build_tracer
from stages.stage0_guardrails import run_guardrails
from stages.stage1_reformulate import reformulate
from stages.stage2_decompose import decompose
from stages.stage25_request_type import classify_request_type
from stages.stage275_domain_inference import infer_domain
from stages.stage3_classify import process_sub_queries_parallel
from stages.stage4_compose import compose
from stages.stage5_quality import quality_check
from stages.stage6_validate import validate_output
from utils.bm25 import load_bm25
from utils.llm import LLMClient

PROJECT_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_DIR.parent
MANIFEST_PATH = PROJECT_DIR / "corpus_cache" / "cache_manifest.json"
INPUT_CSV = REPO_ROOT / "support_tickets" / "support_tickets.csv"
OUTPUT_CSV = REPO_ROOT / "support_tickets" / "output.csv"
TAXONOMY_DIR = PROJECT_DIR / "corpus_cache" / "taxonomy"
ENRICHED_DIR = PROJECT_DIR / "corpus_cache" / "enriched"
INDEXES_DIR = PROJECT_DIR / "corpus_cache" / "indexes"
logger = logging.getLogger(__name__)

COMPANY_ALIASES: dict[str, str | None] = {
    "hackerrank": "HackerRank",
    "hacker rank": "HackerRank",
    "claude": "Claude",
    "anthropic": "Claude",
    "visa": "Visa",
    "none": None,
    "": None,
}


def _normalize_company(raw: str) -> str | None:
    value = raw.strip()
    return COMPANY_ALIASES.get(value.lower(), value or None)


def _short_invalid_response(reason: str, failed_guard: str | None = None) -> OutputRow:
    justification = f"Failed guard: {failed_guard}" if failed_guard else reason
    return OutputRow(
        status=Status.REPLIED,
        product_area="",
        response="Your request cannot be processed as it violates our support policies.",
        justification=justification or "Invalid request.",
        request_type=RequestType.INVALID,
    )


def _error(stage: str, exc: Exception) -> OutputRow:
    return OutputRow(
        status=Status.ERROR,
        product_area="",
        response="A system error occurred. This ticket has been flagged for operator review.",
        justification=f"Infrastructure error at {stage}: {type(exc).__name__}: {exc}",
        request_type=RequestType.PRODUCT_ISSUE,
    )


def _short_out_of_scope_response(reason: str) -> OutputRow:
    return OutputRow(
        status=Status.REPLIED,
        product_area="",
        response="This request falls outside the scope of our support domains.",
        justification=reason or "No matching domain or supporting documents found.",
        request_type=RequestType.INVALID,
    )


def _configure_logging() -> None:
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )


def _escalate(reason: str, request_type: RequestType = RequestType.PRODUCT_ISSUE) -> OutputRow:
    return OutputRow(
        status=Status.ESCALATED,
        product_area="none",
        response="Escalated due to processing failure.",
        justification=reason,
        request_type=request_type,
    )


def load_artifacts() -> tuple[UnifiedTaxonomy, dict[str, MasterIndexEntry], dict[str, Any]]:
    logger.info("Loading artifacts from %s", PROJECT_DIR / "corpus_cache")
    unified = load_unified_taxonomy(TAXONOMY_DIR)
    master = load_master_index(ENRICHED_DIR)
    indexes: dict[str, Any] = {}

    for name in ("hackerrank", "claude", "visa", "all"):
        path = Path(INDEXES_DIR) / f"{name}_bm25.pkl"
        if path.exists():
            indexes[name] = load_bm25(str(path))

    return unified, master, indexes


def _select_index_for_ticket(
    ticket: Ticket,
    reformulated: str,
    unified: UnifiedTaxonomy,
    indexes: dict[str, Any],
) -> tuple[Any, list[str]] | None:
    domain = (ticket.company or "").lower()
    if domain in indexes:
        return indexes[domain]

    if ticket.company is not None:
        return indexes.get("all")

    all_index = indexes.get("all")
    if all_index is None:
        return None

    inference = infer_domain(reformulated, unified, {"all": all_index})
    if inference.short_circuit:
        return None

    if inference.inferred_domain and inference.inferred_domain in indexes:
        return indexes[inference.inferred_domain]

    if inference.is_cross_domain:
        for candidate in inference.cross_domain_indexes:
            if candidate in indexes:
                return indexes[candidate]

    return all_index


@_lf_observe(name="process-ticket", capture_input=False, capture_output=False)
def process_ticket(
    ticket: Ticket,
    unified: UnifiedTaxonomy,
    master: dict[str, MasterIndexEntry],
    indexes: dict[str, Any],
    llm: LLMClient,
) -> OutputRow:
    _lf_client().update_current_span(
        input={"issue": ticket.issue[:200], "company": ticket.company},
        metadata={"company": ticket.company or "unknown"},
    )
    labels = list(unified.bare_label_set())

    try:
        guard = run_guardrails(ticket.issue, ticket.subject, llm)
        if not guard.passed:
            return _short_invalid_response(guard.reason or "", guard.failed_guard)

        reformulated = reformulate(ticket.issue, ticket.subject, llm)
        decomposition = decompose(reformulated, ticket.company, labels, llm)
        request_type = classify_request_type(
            reformulated,
            [item.query for item in decomposition.sub_queries],
            ticket.company,
            llm,
        )

        if request_type == RequestType.INVALID:
            return _short_out_of_scope_response("Classified as invalid request type.")

        index_payload = _select_index_for_ticket(ticket, reformulated, unified, indexes)
        if index_payload is None:
            return _short_out_of_scope_response("No relevant domain/index match for ticket.")
        bm25_index, bm25_doc_ids = index_payload

        sub_results = process_sub_queries_parallel(
            decomposition.sub_queries,
            bm25_index,
            bm25_doc_ids,
            master,
            request_type,
            decomposition.shared_context.situational_frame,
            llm,
        )
        output = compose(sub_results, decomposition.shared_context, request_type, unified.bare_label_set(), llm)
        output = quality_check(output, sub_results, llm)
        return validate_output(output, unified.bare_label_set())

    except PipelineError as exc:
        logger.error("Pipeline error stage=%s cause=%s: %s", exc.stage, type(exc.cause).__name__, exc.cause)
        return _error(exc.stage, exc.cause)


def main() -> None:
    _configure_logging()
    if not MANIFEST_PATH.exists():
        logger.error("Corpus cache not found at %s", MANIFEST_PATH)
        print("Corpus cache not found. Build it first:")
        print("  uv run python code/build_corpus.py")
        sys.exit(1)

    if not INPUT_CSV.exists():
        logger.error("Input CSV missing: %s", INPUT_CSV)
        print(f"Input CSV missing: {INPUT_CSV}")
        sys.exit(1)

    logger.info("Loading corpus artifacts")
    unified, master, indexes = load_artifacts()
    llm = LLMClient()
    registry = PromptRegistry()
    tracer = build_tracer()
    logger.info(
        "Loaded artifacts docs=%d labels=%d indexes=%d",
        len(master),
        len(unified.labels),
        len(indexes),
    )

    tickets: list[Ticket] = []
    with INPUT_CSV.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            tickets.append(
                Ticket(
                    issue=row.get("Issue", row.get("issue", "")),
                    subject=row.get("Subject", row.get("subject", "")),
                    company=_normalize_company(row.get("Company", row.get("company", "")) or ""),
                )
            )

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "issue",
            "subject",
            "company",
            "response",
            "product_area",
            "status",
            "request_type",
            "justification",
            "prompt_versions",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for index, ticket in enumerate(tickets, start=1):
            logger.info("Processing ticket %d/%d company=%s subject=%r", index, len(tickets), ticket.company, ticket.subject)
            registry.reset_log()
            try:
                result = process_ticket(ticket, unified, master, indexes, llm)
            except Exception as exc:
                logger.exception("Unexpected pipeline error for ticket %d", index)
                result = _error("unknown", exc)
            finally:
                result.prompt_versions = registry.get_log()

            writer.writerow(
                {
                    "issue": ticket.issue,
                    "subject": ticket.subject,
                    "company": ticket.company or "None",
                    "response": result.response,
                    "product_area": result.product_area,
                    "status": result.status.value,
                    "request_type": result.request_type.value,
                    "justification": result.justification,
                    "prompt_versions": json.dumps(result.prompt_versions),
                }
            )

    _lf_client().flush()
    logger.info("Done. Output written to %s", OUTPUT_CSV)


if __name__ == "__main__":
    main()
