from __future__ import annotations

import json
import re
from pathlib import Path

from schemas import DomainTaxonomy, NormalizedDoc, TaxonomyLabel
from utils.llm import LLMClient
from utils.prompt_registry import PromptRegistry

_LABEL_RE = re.compile(r"^[a-z][a-z0-9_-]+$")
_REGISTRY = PromptRegistry()


def _strip_code_fence(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
    return raw.strip()


def build_taxonomy(
    domain: str,
    docs: list[NormalizedDoc],
    llm: LLMClient,
    required_labels: list[str] | None = None,
) -> DomainTaxonomy:
    doc_summaries = "\n".join(
        f"[{doc.doc_id}] {doc.title}: {doc.first_paragraph[:120]}" for doc in docs
    )
    templates, _ = _REGISTRY.get("corpus_taxonomy")
    if required_labels:
        required_labels_section = templates["required_labels_section"].format(
            required_labels=", ".join(f'"{label}"' for label in required_labels)
        )
    else:
        required_labels_section = ""

    prompt = templates["template"].format(
        domain=domain,
        total_docs=len(docs),
        doc_summaries=doc_summaries,
        required_labels_section=required_labels_section,
    )
    raw = llm.call_r1([{"role": "user", "content": prompt}])
    data = json.loads(_strip_code_fence(raw))
    labels = [TaxonomyLabel(**item) for item in data["labels"]]
    return DomainTaxonomy(domain=domain, overview=data["overview"], labels=labels)


def validate_taxonomy(
    taxonomy: DomainTaxonomy,
    docs: list[NormalizedDoc],
    required_labels: list[str] | None = None,
) -> dict:
    doc_id_set = {doc.doc_id for doc in docs}
    total_docs = len(docs)
    approx_total = sum(label.approximate_doc_count for label in taxonomy.labels)
    if total_docs == 0:
        doc_counts_match = approx_total == 0
    else:
        doc_counts_match = abs(approx_total - total_docs) / total_docs <= 0.10

    label_names = [label.product_area_label for label in taxonomy.labels]
    no_duplicates = len(label_names) == len(set(label_names))
    format_ok = all(_LABEL_RE.match(label_name) for label_name in label_names)
    example_ids_valid = all(
        example_id in doc_id_set
        for label in taxonomy.labels
        for example_id in label.example_doc_ids
    )
    required_labels_present = all(
        req in label_names for req in (required_labels or [])
    )

    passed = doc_counts_match and no_duplicates and format_ok and example_ids_valid and required_labels_present
    return {
        "passed": passed,
        "details": {
            "doc_counts_match": doc_counts_match,
            "example_ids_valid": example_ids_valid,
            "label_format_compliant": format_ok,
            "no_duplicate_labels": no_duplicates,
            "required_labels_present": required_labels_present,
        },
    }


def save_taxonomy(taxonomy: DomainTaxonomy, out_dir: str) -> None:
    path = Path(out_dir) / f"{taxonomy.domain}_taxonomy.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(taxonomy.model_dump_json(indent=2), encoding="utf-8")


def load_taxonomy(domain: str, taxonomy_dir: str) -> DomainTaxonomy:
    path = Path(taxonomy_dir) / f"{domain}_taxonomy.json"
    return DomainTaxonomy.model_validate_json(path.read_text(encoding="utf-8"))
