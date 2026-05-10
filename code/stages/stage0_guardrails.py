from __future__ import annotations

import logging
from dataclasses import dataclass

import litellm
from guardrails import Guard, OnFailAction, Validator, register_validator
from guardrails.errors import ValidationError
from guardrails.validator_base import FailResult, PassResult

from config import get_settings
from exceptions import GuardrailError
from utils.llm import LLMClient
from utils.prompt_registry import PromptRegistry
from utils.stage_handler import stage_handler

litellm.suppress_debug_info = True
litellm.set_verbose = False

logger = logging.getLogger(__name__)

_MODEL = "deepseek/deepseek-chat"
_REGISTRY = PromptRegistry()
_GUARD_TEMPLATES, _ = _REGISTRY.get("stage0_guardrails")


def _llm_binary(prompt: str) -> str:
    """Call deepseek-chat via LiteLLM and return the stripped lowercase reply."""
    response = litellm.completion(
        model=_MODEL,
        messages=[{"role": "user", "content": prompt}],
        api_key=get_settings().deepseek_api_key,
        temperature=0,
        max_tokens=5,
    )
    return response.choices[0].message.content.strip().lower()


@dataclass
class GuardResult:
    passed: bool
    failed_guard: str | None = None
    reason: str | None = None


@register_validator(name="support-triage-prompt-injection", data_type="string")
class PromptInjectionValidator(Validator):
    def validate(self, value: str, metadata: dict) -> PassResult | FailResult:
        prompt = _GUARD_TEMPLATES["prompt_injection"].format(value=value)
        try:
            answer = _llm_binary(prompt)
        except Exception:
            logger.warning("Prompt injection guard LLM call failed; defaulting to pass.")
            return PassResult()

        if answer.startswith("yes"):
            return FailResult(error_message="Prompt injection detected.")
        return PassResult()


@register_validator(name="support-triage-malicious-intent", data_type="string")
class MaliciousIntentValidator(Validator):
    def validate(self, value: str, metadata: dict) -> PassResult | FailResult:
        prompt = _GUARD_TEMPLATES["malicious_intent"].format(value=value)
        try:
            answer = _llm_binary(prompt)
        except Exception:
            logger.warning("Malicious intent guard LLM call failed; defaulting to pass.")
            return PassResult()

        if answer.startswith("yes"):
            return FailResult(error_message="Malicious intent detected.")
        return PassResult()


@register_validator(name="support-triage-support-scope", data_type="string")
class SupportScopeValidator(Validator):
    def validate(self, value: str, metadata: dict) -> PassResult | FailResult:
        prompt = _GUARD_TEMPLATES["support_scope"].format(value=value)
        try:
            answer = _llm_binary(prompt)
        except Exception:
            logger.warning("Support scope guard LLM call failed; defaulting to pass.")
            return PassResult()

        if answer.startswith("no"):
            return FailResult(error_message="Ticket is outside supported support domains.")
        return PassResult()


_injection_guard = Guard().use(PromptInjectionValidator(on_fail=OnFailAction.EXCEPTION))
_malicious_guard = Guard().use(MaliciousIntentValidator(on_fail=OnFailAction.EXCEPTION))
_scope_guard = Guard().use(SupportScopeValidator(on_fail=OnFailAction.EXCEPTION))


@stage_handler("stage0_guardrails", maps={Exception: GuardrailError})
def run_guardrails(issue: str, subject: str, llm: LLMClient | None) -> GuardResult:
    _ = llm
    text = f"{subject} {issue}"

    for guard, guard_name in (
        (_injection_guard, "prompt_injection"),
        (_malicious_guard, "malicious_intent"),
        (_scope_guard, "support_scope"),
    ):
        try:
            guard.validate(text)
        except ValidationError as exc:
            reason = str(exc.args[0]).removeprefix(
                "Validation failed for field with errors: "
            )
            return GuardResult(passed=False, failed_guard=guard_name, reason=reason)

    return GuardResult(passed=True)
