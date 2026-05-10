from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MANIFEST_PATH = Path(__file__).resolve().parent.parent / "corpus_cache" / "cache_manifest.json"

_STEP_TO_DURATION_KEY = {
    "step1_normalize": "step1",
    "step2_taxonomy": "step2",
    "step3_enrich": "step3",
    "step4_reconcile": "step4",
    "step5_master_index": "step5",
    "step6_bm25": "step6",
}


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).astimezone().isoformat()


def _default_manifest() -> dict[str, Any]:
    return {
        "version": "1.0.0",
        "build_start": _now_iso(),
        "build_end": None,
        "build_timestamps": {},
        "build_duration_seconds": {},
        "llm_config": {},
        "steps_completed": {step_key: False for step_key in _STEP_TO_DURATION_KEY},
        "corpus_status": {},
        "taxonomy_stats": {},
        "quality_gates": {},
        "indexes": {},
    }


def load_manifest() -> dict[str, Any]:
    if MANIFEST_PATH.exists():
        loaded = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        manifest = _default_manifest()
        manifest.update(loaded)
        default_steps = _default_manifest()["steps_completed"]
        manifest["steps_completed"] = {
            **default_steps,
            **loaded.get("steps_completed", {}),
        }
        return manifest
    return _default_manifest()


def save_manifest(manifest: dict[str, Any]) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def mark_step_complete(step_key: str, manifest: dict[str, Any], duration_seconds: float) -> None:
    if step_key not in _STEP_TO_DURATION_KEY:
        raise ValueError(f"Unknown step key: {step_key}")

    manifest["steps_completed"][step_key] = True
    manifest["build_timestamps"][step_key] = _now_iso()
    duration_key = _STEP_TO_DURATION_KEY[step_key]
    manifest["build_duration_seconds"][duration_key] = round(float(duration_seconds), 1)
    save_manifest(manifest)
