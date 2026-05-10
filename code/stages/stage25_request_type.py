from __future__ import annotations

import json

from exceptions import JSONParseError, ProviderError
from schemas import RequestType
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


@stage_handler("stage25_request_type", maps={json.JSONDecodeError: JSONParseError, AllProvidersFailedError: ProviderError})
def classify_request_type(
    reformulated_query: str,
    sub_queries: list[str],
    company: str | None,
    llm: LLMClient,
) -> RequestType:
    templates, _ = _REGISTRY.get("stage25_request_type")
    prompt = templates["template"].format(
        query=reformulated_query,
        sub_queries="; ".join(sub_queries),
        company=company or "Unknown",
    )
    raw = llm.call_v3([{"role": "user", "content": prompt}], json_mode=True)
    data = _parse_json_with_fallback(raw)
    return RequestType(data["request_type"])
