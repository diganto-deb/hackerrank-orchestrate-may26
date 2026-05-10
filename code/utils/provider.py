from __future__ import annotations

import logging
import time
from typing import Callable

from langfuse import observe as _lf_observe, get_client as _lf_client
from openai import OpenAI, APIError as OpenAIAPIError, RateLimitError as OpenAIRateLimitError

from config import get_settings

logger = logging.getLogger(__name__)

# Groq hosts openai/gpt-oss-120b — same model for both tiers, reasoning_effort varies
_CHAT_MODELS: dict[str, str] = {
    "deepseek": "deepseek-chat",
    "claude": "claude-haiku-4-5-20251001",
    "groq": "openai/gpt-oss-120b",
}
_REASONER_MODELS: dict[str, str] = {
    "deepseek": "deepseek-reasoner",
    "claude": "claude-sonnet-4-6",
    "groq": "openai/gpt-oss-120b",
}
_MAX_RETRIES = 3


class AllProvidersFailedError(RuntimeError):
    pass


@_lf_observe(name="deepseek-generation", as_type="generation", capture_input=False, capture_output=False)
def _deepseek_call(
    messages: list[dict],
    model: str,
    json_mode: bool = False,
) -> str:
    settings = get_settings()
    client = OpenAI(api_key=settings.deepseek_api_key, base_url="https://api.deepseek.com", timeout=120)
    kwargs: dict = {}
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    resp = client.chat.completions.create(
        model=model, messages=messages, temperature=0, **kwargs
    )
    content = resp.choices[0].message.content
    _lf_client().update_current_generation(
        model=model,
        input=messages,
        output=content,
        usage_details={
            "prompt_tokens": resp.usage.prompt_tokens,
            "completion_tokens": resp.usage.completion_tokens,
        },
    )
    return content


@_lf_observe(name="claude-generation", as_type="generation", capture_input=False, capture_output=False)
def _claude_call(
    messages: list[dict],
    model: str,
    json_mode: bool = False,
) -> str:
    import anthropic

    settings = get_settings()
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    user_messages = [m for m in messages if m["role"] != "system"]
    system_messages = [m["content"] for m in messages if m["role"] == "system"]
    system = system_messages[0] if system_messages else anthropic.NOT_GIVEN
    resp = client.messages.create(
        model=model,
        max_tokens=4096,
        system=system,
        messages=user_messages,
        temperature=0,
    )
    content = resp.content[0].text
    _lf_client().update_current_generation(
        model=model,
        input=user_messages,
        output=content,
        usage_details={
            "prompt_tokens": resp.usage.input_tokens,
            "completion_tokens": resp.usage.output_tokens,
        },
    )
    return content


@_lf_observe(name="groq-generation", as_type="generation", capture_input=False, capture_output=False)
def _groq_call(
    messages: list[dict],
    model: str,
    json_mode: bool = False,
    reasoning_effort: str = "medium",
) -> str:
    from groq import Groq

    settings = get_settings()
    client = Groq(api_key=settings.groq_api_key)
    kwargs: dict = {"reasoning_effort": reasoning_effort}
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=1,
        max_completion_tokens=8192,
        top_p=1,
        stream=False,
        stop=None,
        **kwargs,
    )
    content = resp.choices[0].message.content
    _lf_client().update_current_generation(
        model=model,
        input=messages,
        output=content,
        usage_details={
            "prompt_tokens": resp.usage.prompt_tokens,
            "completion_tokens": resp.usage.completion_tokens,
        },
    )
    return content


def _groq_chat_call(messages: list[dict], model: str, json_mode: bool = False) -> str:
    return _groq_call(messages, model, json_mode=json_mode, reasoning_effort="low")


def _groq_reasoner_call(messages: list[dict], model: str, json_mode: bool = False) -> str:
    return _groq_call(messages, model, json_mode=json_mode, reasoning_effort="medium")


def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, (OpenAIRateLimitError, OpenAIAPIError)):
        return True
    try:
        import anthropic
        if isinstance(exc, (anthropic.RateLimitError, anthropic.APIError)):
            return True
    except ImportError:
        pass
    try:
        from groq import RateLimitError as GroqRateLimitError, APIError as GroqAPIError
        if isinstance(exc, (GroqRateLimitError, GroqAPIError)):
            return True
    except ImportError:
        pass
    return False


def _call_with_retry(fn: Callable, *args, **kwargs) -> str:
    delay = 1.0
    last_exc: Exception | None = None
    for attempt in range(_MAX_RETRIES):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            if not _is_retryable(exc):
                raise
            last_exc = exc
            if attempt < _MAX_RETRIES - 1:
                logger.warning(
                    "Provider call failed (attempt %d/%d): %s",
                    attempt + 1,
                    _MAX_RETRIES,
                    exc,
                )
                time.sleep(delay)
                delay *= 2
    raise last_exc  # type: ignore[misc]


class ProviderRouter:
    def __init__(self) -> None:
        settings = get_settings()
        self._providers: dict[str, list[tuple[str, Callable]]] = {
            "chat": [("deepseek", _deepseek_call)],
            "reasoner": [("deepseek", _deepseek_call)],
        }
        if settings.anthropic_api_key:
            self._providers["chat"].append(("claude", _claude_call))
            self._providers["reasoner"].append(("claude", _claude_call))
        if settings.groq_api_key:
            self._providers["chat"].append(("groq", _groq_chat_call))
            self._providers["reasoner"].append(("groq", _groq_reasoner_call))

    def call(
        self,
        messages: list[dict],
        model_tier: str,
        json_mode: bool = False,
    ) -> str:
        models = _CHAT_MODELS if model_tier == "chat" else _REASONER_MODELS
        provider_chain = self._providers.get(model_tier, self._providers["chat"])
        errors: list[Exception] = []
        for provider_name, call_fn in provider_chain:
            model = models[provider_name]
            try:
                return _call_with_retry(call_fn, messages, model, json_mode)
            except Exception as exc:
                logger.warning(
                    "Provider %s exhausted retries: %s", provider_name, exc
                )
                errors.append(exc)
        raise AllProvidersFailedError(f"All providers failed. Errors: {errors}")
