from __future__ import annotations

import logging
import re

from utils.provider import ProviderRouter

logger = logging.getLogger(__name__)

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _strip_think_blocks(text: str) -> str:
    return _THINK_RE.sub("", text).strip()


class LLMUsage:
    def __init__(
        self,
        v3_calls: int = 0,
        v3_prompt_tokens: int = 0,
        v3_completion_tokens: int = 0,
        r1_calls: int = 0,
        r1_prompt_tokens: int = 0,
        r1_completion_tokens: int = 0,
    ) -> None:
        self.v3_calls = v3_calls
        self.v3_prompt_tokens = v3_prompt_tokens
        self.v3_completion_tokens = v3_completion_tokens
        self.r1_calls = r1_calls
        self.r1_prompt_tokens = r1_prompt_tokens
        self.r1_completion_tokens = r1_completion_tokens

    @classmethod
    def from_dict(cls, data: dict | None) -> "LLMUsage":
        data = data or {}
        v3 = data.get("v3", {})
        r1 = data.get("r1", {})
        return cls(
            v3_calls=int(v3.get("total_calls", 0)),
            v3_prompt_tokens=int(v3.get("total_prompt_tokens", 0)),
            v3_completion_tokens=int(v3.get("total_completion_tokens", 0)),
            r1_calls=int(r1.get("total_calls", 0)),
            r1_prompt_tokens=int(r1.get("total_prompt_tokens", 0)),
            r1_completion_tokens=int(r1.get("total_completion_tokens", 0)),
        )

    def to_dict(self) -> dict:
        return {
            "v3": {
                "model": "deepseek-chat",
                "temperature": 0,
                "total_calls": self.v3_calls,
                "total_prompt_tokens": self.v3_prompt_tokens,
                "total_completion_tokens": self.v3_completion_tokens,
            },
            "r1": {
                "model": "deepseek-reasoner",
                "temperature": 0,
                "total_calls": self.r1_calls,
                "total_prompt_tokens": self.r1_prompt_tokens,
                "total_completion_tokens": self.r1_completion_tokens,
            },
        }


class LLMClient:
    def __init__(self, usage: LLMUsage | None = None) -> None:
        self._router = ProviderRouter()
        self.usage = usage or LLMUsage()

    def call_v3(self, messages: list[dict], json_mode: bool = False) -> str:
        content = self._router.call(messages, "chat", json_mode=json_mode)
        self.usage.v3_calls += 1
        logger.info("LLM call tier=chat")
        return content

    def call_r1(self, messages: list[dict]) -> str:
        content = self._router.call(messages, "reasoner")
        self.usage.r1_calls += 1
        logger.info("LLM call tier=reasoner")
        return _strip_think_blocks(content)
