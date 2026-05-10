from __future__ import annotations

import logging
from typing import Any

from config import get_settings

logger = logging.getLogger(__name__)


class NullSpan:
    def span(self, name: str, **kwargs: Any) -> "NullSpan":
        return NullSpan()

    def record(self, **kwargs: Any) -> None:
        pass

    def end(self, **kwargs: Any) -> None:
        pass


class NullTracer:
    def trace(self, name: str, **kwargs: Any) -> NullSpan:
        return NullSpan()

    def flush(self) -> None:
        pass


class LangFuseSpan:
    def __init__(self, lf_span: Any) -> None:
        self._span = lf_span

    def span(self, name: str, **kwargs: Any) -> "LangFuseSpan":
        child = self._span.span(name=name, **kwargs)
        return LangFuseSpan(child)

    def record(self, **kwargs: Any) -> None:
        try:
            self._span.update(**kwargs)
        except Exception as exc:
            logger.debug("LangFuse record error: %s", exc)

    def end(self, **kwargs: Any) -> None:
        try:
            self._span.end(**kwargs)
        except Exception as exc:
            logger.debug("LangFuse end error: %s", exc)


class LangFuseTracer:
    def __init__(self, client: Any) -> None:
        self._client = client

    def trace(self, name: str, **kwargs: Any) -> LangFuseSpan:
        root = self._client.trace(name=name, **kwargs)
        return LangFuseSpan(root)

    def flush(self) -> None:
        try:
            self._client.flush()
        except Exception as exc:
            logger.debug("LangFuse flush error: %s", exc)


def build_tracer() -> NullTracer | LangFuseTracer:
    settings = get_settings()
    if settings.langfuse_secret_key and settings.langfuse_public_key:
        try:
            from langfuse import Langfuse
            client = Langfuse(
                secret_key=settings.langfuse_secret_key,
                public_key=settings.langfuse_public_key,
                base_url=settings.langfuse_base_url,
            )
            logger.info("LangFuse tracer initialised at %s", settings.langfuse_base_url)
            return LangFuseTracer(client)
        except Exception as exc:
            logger.warning("Failed to init LangFuse; using NullTracer: %s", exc)
    return NullTracer()
