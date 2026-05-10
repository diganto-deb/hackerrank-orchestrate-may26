from __future__ import annotations

from exceptions import ProviderError
from utils.llm import LLMClient
from utils.prompt_registry import PromptRegistry
from utils.provider import AllProvidersFailedError
from utils.stage_handler import stage_handler

_REGISTRY = PromptRegistry()


@stage_handler("stage1_reformulate", maps={AllProvidersFailedError: ProviderError})
def reformulate(issue: str, subject: str, llm: LLMClient) -> str:
    templates, _ = _REGISTRY.get("stage1_reformulate")
    prompt = templates["template"].format(subject=subject or "(none)", issue=issue)
    return llm.call_v3([{"role": "user", "content": prompt}]).strip()
