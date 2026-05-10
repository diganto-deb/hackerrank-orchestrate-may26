from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from schemas import DomainTaxonomy, UnifiedLabel, UnifiedTaxonomy
from utils.llm import LLMClient
from utils.prompt_registry import PromptRegistry

logger = logging.getLogger(__name__)
_REGISTRY = PromptRegistry()


def prefixed_label(domain: str, bare_label: str) -> str:
    return f"{domain}:{bare_label}"


def _normalize_label_from_domain_taxonomy(domain: str, tax: DomainTaxonomy) -> list[UnifiedLabel]:
    labels: list[UnifiedLabel] = []
    for item in tax.labels:
        bare = item.product_area_label
        labels.append(
            UnifiedLabel(
                domain=domain,
                bare_label=bare,
                prefixed_label=prefixed_label(domain, bare),
                description=item.description,
                doc_count=item.approximate_doc_count,
            )
        )
    return labels


def reconcile_taxonomies_deterministic(
    taxonomies: dict[str, DomainTaxonomy],
) -> UnifiedTaxonomy:
    labels: list[UnifiedLabel] = []
    for domain in sorted(taxonomies):
        labels.extend(_normalize_label_from_domain_taxonomy(domain, taxonomies[domain]))
    labels.sort(key=lambda value: (value.domain, value.bare_label, value.description))
    return UnifiedTaxonomy(labels=labels)


def _strip_fences(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    return text.strip()


def _extract_json_array(raw: str) -> list[dict]:
    text = _strip_fences(raw)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1 and end > start:
            candidate = text[start : end + 1]
            return json.loads(candidate)
        raise


def reconcile_taxonomies(
    taxonomies: dict[str, DomainTaxonomy],
    llm: LLMClient,
) -> UnifiedTaxonomy:
    tax_text = ""
    for domain in sorted(taxonomies):
        tax = taxonomies[domain]
        tax_text += f"\nDomain: {domain}\n"
        for lbl in tax.labels:
            tax_text += (
                f"  - {lbl.product_area_label} "
                f"({lbl.approximate_doc_count} docs): {lbl.description}\n"
            )

    templates, _ = _REGISTRY.get("corpus_reconcile")
    raw = llm.call_r1([
        {"role": "user", "content": templates["template"].format(taxonomies=tax_text)}
    ])
    try:
        data = _extract_json_array(raw)
        labels = [UnifiedLabel(**item) for item in data]
        labels.sort(key=lambda value: (value.domain, value.bare_label, value.description))
        return UnifiedTaxonomy(labels=labels)
    except (json.JSONDecodeError, TypeError, ValueError, KeyError) as exc:
        logger.warning(
            "Reconcile LLM output was not valid JSON; falling back to deterministic reconciliation. excerpt=%r error=%s",
            raw[:500],
            exc,
        )
        return reconcile_taxonomies_deterministic(taxonomies)


def save_unified_taxonomy(unified: UnifiedTaxonomy, taxonomy_dir: str) -> None:
    path = Path(taxonomy_dir) / "unified_taxonomy.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(unified.model_dump_json(indent=2), encoding="utf-8")


def load_unified_taxonomy(taxonomy_dir: str) -> UnifiedTaxonomy:
    path = Path(taxonomy_dir) / "unified_taxonomy.json"
    return UnifiedTaxonomy.model_validate_json(path.read_text(encoding="utf-8"))
