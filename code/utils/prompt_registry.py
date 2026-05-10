from __future__ import annotations

from pathlib import Path

import yaml

_DEFAULT_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


class PromptRegistry:
    def __init__(self, prompts_dir: Path | None = None) -> None:
        self._dir = prompts_dir or _DEFAULT_PROMPTS_DIR
        self._cache: dict[str, tuple[dict[str, str], str]] = {}
        self._log: dict[str, str] = {}

    def get(
        self, stage: str, version: str = "latest"
    ) -> tuple[dict[str, str], str]:
        cache_key = f"{stage}@{version}"
        if cache_key in self._cache:
            templates, version_string = self._cache[cache_key]
            self._log[stage] = version_string
            return templates, version_string

        stage_dir = self._dir / stage
        if not stage_dir.is_dir():
            raise FileNotFoundError(
                f"No prompt directory found for stage '{stage}' at {stage_dir}"
            )

        if version == "latest":
            yaml_files = sorted(stage_dir.glob("v*.yaml"))
            if not yaml_files:
                raise FileNotFoundError(f"No YAML files found in {stage_dir}")
            yaml_path = yaml_files[-1]
        else:
            yaml_path = stage_dir / f"v{version}.yaml"
            if not yaml_path.exists():
                raise FileNotFoundError(
                    f"Prompt version {version} not found at {yaml_path}"
                )

        with yaml_path.open(encoding="utf-8") as handle:
            data = yaml.safe_load(handle)

        resolved_version = str(data.get("version", yaml_path.stem.lstrip("v")))
        version_string = f"{stage}@v{resolved_version}"
        templates = {
            k: v
            for k, v in data.items()
            if k not in ("version", "stage") and isinstance(v, str)
        }

        self._cache[cache_key] = (templates, version_string)
        self._log[stage] = version_string
        return templates, version_string

    def reset_log(self) -> None:
        self._log.clear()

    def get_log(self) -> dict[str, str]:
        return dict(self._log)
