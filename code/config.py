from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

_REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_REPO_ROOT / ".env")
# Langfuse v4 reads LANGFUSE_BASE_URL; map LANGFUSE_HOST for shells that export the old name
if os.environ.get("LANGFUSE_HOST") and not os.environ.get("LANGFUSE_BASE_URL"):
    os.environ["LANGFUSE_BASE_URL"] = os.environ["LANGFUSE_HOST"]


class Settings(BaseSettings):
    deepseek_api_key: str
    anthropic_api_key: str | None = None
    groq_api_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_public_key: str | None = None
    langfuse_base_url: str = "https://cloud.langfuse.com"

    model_config = SettingsConfigDict(
        env_file=(
            Path(__file__).resolve().parent / ".env",
            _REPO_ROOT / ".env",
        ),
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    try:
        return Settings()
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Missing DEEPSEEK_API_KEY. Set it in repo-root `.env`."
        ) from exc
