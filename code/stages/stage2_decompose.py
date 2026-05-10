from __future__ import annotations

import json

from exceptions import JSONParseError, ProviderError
from schemas import DecompositionResult, SharedContext, SubQuery
from utils.llm import LLMClient
from utils.prompt_registry import PromptRegistry
from utils.provider import AllProvidersFailedError
from utils.stage_handler import stage_handler

_REGISTRY = PromptRegistry()


def _parse_json_with_fallback(raw: str) -> dict:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(raw[start : end + 1])


@stage_handler("stage2_decompose", maps={json.JSONDecodeError: JSONParseError, AllProvidersFailedError: ProviderError})
def decompose(
    reformulated_query: str,
    company: str | None,
    taxonomy_labels: list[str],
    llm: LLMClient,
) -> DecompositionResult:
    templates, _ = _REGISTRY.get("stage2_decompose")
    taxonomy_summary = ", ".join(taxonomy_labels[:30])
    prompt = templates["template"].format(
        company=company or "Unknown",
        taxonomy_summary=taxonomy_summary,
        query=reformulated_query,
    )
    raw = llm.call_v3([{"role": "user", "content": prompt}], json_mode=True)
    data = _parse_json_with_fallback(raw)
    return DecompositionResult(
        shared_context=SharedContext(**data["shared_context"]),
        sub_queries=[SubQuery(**sub_query) for sub_query in data["sub_queries"]],
    )
