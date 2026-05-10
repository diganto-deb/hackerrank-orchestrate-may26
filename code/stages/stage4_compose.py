from __future__ import annotations

import json

from exceptions import JSONParseError, ProviderError
from schemas import (
    OutputRow,
    RequestType,
    SharedContext,
    SubQueryClassification,
    SubQueryResult,
)
from utils.llm import LLMClient
from utils.prompt_registry import PromptRegistry
from utils.provider import AllProvidersFailedError
from utils.stage_handler import stage_handler

_REGISTRY = PromptRegistry()


def _bare_label(label: str) -> str:
    if ":" in label:
        return label.split(":", 1)[1]
    return label


def _format_sub_results(results: list[SubQueryResult]) -> str:
    lines: list[str] = []
    for result in results:
        doc_ids = [doc.doc_id for doc in result.retrieved_docs]
        doc_summary = "\n".join(
            f"  [{doc.doc_id}] ({doc.product_area}) {doc.title}: {doc.body_clean[:400]}"
            for doc in result.retrieved_docs[:2]
        )
        lines.append(
            f"Q{result.sub_query_id}: {result.query}\n"
            f"  Classification: {result.classification.value}\n"
            f"  Docs: {doc_ids}\n"
            f"{doc_summary}"
        )
    return "\n\n".join(lines)


def _default_product_area(sub_results: list[SubQueryResult]) -> str:
    for result in sub_results:
        if result.retrieved_docs:
            return _bare_label(result.retrieved_docs[0].product_area)
    return "none"


@stage_handler("stage4_compose", maps={json.JSONDecodeError: JSONParseError, AllProvidersFailedError: ProviderError})
def compose(
    sub_results: list[SubQueryResult],
    shared_context: SharedContext,
    request_type: RequestType,
    unified_taxonomy_labels: set[str],
    llm: LLMClient,
) -> OutputRow:
    has_answerable = any(
        r.classification == SubQueryClassification.ANSWERABLE for r in sub_results
    )
    status_default = "replied" if has_answerable else "escalated"
    default_product_area = _default_product_area(sub_results)

    templates, _ = _REGISTRY.get("stage4_compose")
    prompt = templates["template"].format(
        situation=shared_context.situational_frame,
        request_type=request_type.value,
        company=shared_context.company or "Unknown",
        sub_results=_format_sub_results(sub_results),
        taxonomy_labels=", ".join(sorted(unified_taxonomy_labels)),
        request_type_value=request_type.value,
    )
    raw = llm.call_v3([{"role": "user", "content": prompt}], json_mode=True)
    data = json.loads(raw)

    status = data.get("status", status_default)
    return OutputRow(
        status=status,
        product_area=_bare_label(data.get("product_area", default_product_area)),
        response=data["response"],
        justification=data["justification"],
        request_type=request_type,
    )
