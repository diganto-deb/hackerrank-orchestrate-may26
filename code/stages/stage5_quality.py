from __future__ import annotations

import json

from exceptions import JSONParseError, ProviderError
from schemas import OutputRow, SubQueryResult
from utils.llm import LLMClient
from utils.prompt_registry import PromptRegistry
from utils.provider import AllProvidersFailedError
from utils.stage_handler import stage_handler

_REGISTRY = PromptRegistry()

_FAILURE_GUIDANCE = {
    "grounding": "Use only claims traceable to listed doc IDs.",
    "completeness": "Address every listed sub-query explicitly.",
    "scope": "Remove advice outside documented support scope.",
    "coherence": "Align status, product_area, justification, and request_type.",
    "contradiction": "Remove any internal inconsistencies.",
}


def _collect_sub_queries(sub_results: list[SubQueryResult]) -> list[str]:
    return [result.query for result in sub_results]


def _collect_doc_ids(sub_results: list[SubQueryResult]) -> list[str]:
    doc_ids = {doc.doc_id for result in sub_results for doc in result.retrieved_docs}
    return sorted(doc_ids)


def _bare_label(label: str) -> str:
    if ":" in label:
        return label.split(":", 1)[1]
    return label


@stage_handler("stage5_quality", maps={json.JSONDecodeError: JSONParseError, AllProvidersFailedError: ProviderError})
def quality_check(
    output: OutputRow,
    sub_results: list[SubQueryResult],
    llm: LLMClient,
    max_retries: int = 2,
) -> OutputRow:
    sub_queries = _collect_sub_queries(sub_results)
    doc_ids = _collect_doc_ids(sub_results)
    templates, _ = _REGISTRY.get("stage5_quality")

    current = output
    for attempt in range(max_retries + 1):
        qc_prompt = templates["qc_template"].format(
            sub_queries="; ".join(sub_queries),
            doc_ids=", ".join(doc_ids),
            status=current.status.value,
            product_area=_bare_label(current.product_area),
            response=current.response,
            justification=current.justification,
            request_type=current.request_type.value,
        )
        qc_raw = llm.call_r1([{"role": "user", "content": qc_prompt}])
        qc = json.loads(qc_raw)

        if qc.get("passed", False):
            return current

        if attempt == max_retries:
            return current

        failures = qc.get("failures", [])
        guidance = " ".join(_FAILURE_GUIDANCE.get(failure, "") for failure in failures)
        regen_prompt = templates["regen_template"].format(
            failures=", ".join(failures),
            sub_queries="; ".join(sub_queries),
            doc_ids=", ".join(doc_ids),
            status=current.status.value,
            product_area=_bare_label(current.product_area),
            response=current.response,
            justification=current.justification,
            request_type=current.request_type.value,
            guidance=guidance,
        )
        regen_raw = llm.call_v3(
            [{"role": "user", "content": regen_prompt}], json_mode=True
        )
        data = json.loads(regen_raw)

        current = OutputRow(
            status=data.get("status", current.status.value),
            product_area=_bare_label(data.get("product_area", current.product_area)),
            response=data.get("response", current.response),
            justification=data.get("justification", current.justification),
            request_type=current.request_type,
        )

    return current
