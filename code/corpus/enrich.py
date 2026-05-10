from __future__ import annotations

import json
from pathlib import Path

from schemas import DocType, DomainTaxonomy, EnrichedDoc, NormalizedDoc
from utils.llm import LLMClient
from utils.prompt_registry import PromptRegistry

_REGISTRY = PromptRegistry()


def _labels_block(taxonomy: DomainTaxonomy) -> str:
    return "\n".join(
        f"  - {label.product_area_label}: {label.description}" for label in taxonomy.labels
    )


def enrich_doc(doc: NormalizedDoc, taxonomy: DomainTaxonomy, llm: LLMClient) -> EnrichedDoc:
    templates, _ = _REGISTRY.get("corpus_enrich")
    prompt = templates["template"].format(
        domain=doc.domain,
        labels=_labels_block(taxonomy),
        doc_id=doc.doc_id,
        title=doc.title,
        body=doc.body_clean[:4000],
    )
    raw = llm.call_v3([{"role": "user", "content": prompt}], json_mode=True)
    data = json.loads(raw)

    return EnrichedDoc(
        doc_id=doc.doc_id,
        product_area_primary=data["product_area_primary"],
        doc_type=DocType(data["doc_type"]),
        summary=data["summary"],
        keywords=data["keywords"],
        confidence=float(data.get("confidence", 0.8)),
        cross_refs=data.get("cross_refs", []),
    )


def validate_enrichment(
    enriched: EnrichedDoc,
    taxonomy: DomainTaxonomy,
    valid_labels: set[str],
) -> dict | None:
    issues: list[str] = []

    if enriched.product_area_primary not in valid_labels:
        return {
            "hard_fail": True,
            "reason": f"invented label: {enriched.product_area_primary}",
        }

    if enriched.confidence < 0.7:
        issues.append(f"confidence={enriched.confidence:.2f}")

    if len(enriched.keywords) < 3:
        issues.append(f"only {len(enriched.keywords)} keywords")

    if enriched.summary.startswith("This document"):
        issues.append("summary starts with 'This document'")

    if issues:
        return {"hard_fail": False, "reason": "; ".join(issues)}

    return None


def save_enriched(enriched_list: list[EnrichedDoc], domain: str, out_dir: str) -> None:
    path = Path(out_dir) / f"{domain}_enriched.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([entry.model_dump() for entry in enriched_list], indent=2), encoding="utf-8")


def load_enriched(domain: str, enriched_dir: str) -> list[EnrichedDoc]:
    path = Path(enriched_dir) / f"{domain}_enriched.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return [EnrichedDoc.model_validate(item) for item in data]
