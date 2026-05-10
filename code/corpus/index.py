from __future__ import annotations

import json
from pathlib import Path

from schemas import EnrichedDoc, MasterIndexEntry, NormalizedDoc, UnifiedTaxonomy
from utils.bm25 import build_bm25, save_bm25

DOMAINS = ("hackerrank", "claude", "visa")


def build_master_index(
    normalized: dict[str, list[NormalizedDoc]],
    enriched: dict[str, list[EnrichedDoc]],
    unified: UnifiedTaxonomy,
) -> dict[str, MasterIndexEntry]:
    prefix_map: dict[str, str] = {
        f"{label.domain}:{label.bare_label}": label.prefixed_label for label in unified.labels
    }

    normalized_lookup: dict[str, NormalizedDoc] = {}
    for docs in normalized.values():
        for doc in docs:
            normalized_lookup[doc.doc_id] = doc

    merged: dict[str, MasterIndexEntry] = {}
    for domain in sorted(enriched):
        for doc in enriched[domain]:
            normalized_doc = normalized_lookup.get(doc.doc_id)
            if normalized_doc is None:
                continue
            bare_key = f"{normalized_doc.domain}:{doc.product_area_primary}"
            merged[doc.doc_id] = MasterIndexEntry(
                domain=normalized_doc.domain,
                product_area=prefix_map.get(bare_key, bare_key),
                doc_type=doc.doc_type,
                summary=doc.summary,
                keywords=doc.keywords,
                title=normalized_doc.title,
                source_url=normalized_doc.existing_metadata.get("source_url", ""),
                body_clean=normalized_doc.body_clean,
            )

    return dict(sorted(merged.items(), key=lambda item: item[0]))


def save_master_index(master: dict[str, MasterIndexEntry], enriched_dir: str) -> None:
    path = Path(enriched_dir) / "master_index.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {doc_id: entry.model_dump() for doc_id, entry in sorted(master.items())}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_master_index(enriched_dir: str) -> dict[str, MasterIndexEntry]:
    path = Path(enriched_dir) / "master_index.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return {doc_id: MasterIndexEntry.model_validate(entry) for doc_id, entry in data.items()}


def bm25_source_text(entry: MasterIndexEntry) -> str:
    keywords = " ".join(entry.keywords)
    return (
        f"{entry.title} {entry.product_area} {entry.domain} "
        f"{keywords} {entry.summary} {entry.body_clean}"
    )


def build_bm25_indexes(master: dict[str, MasterIndexEntry], indexes_dir: str) -> None:
    out_dir = Path(indexes_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    docs_by_domain: dict[str, list[tuple[str, str]]] = {domain: [] for domain in DOMAINS}
    all_docs: list[tuple[str, str]] = []

    for doc_id, entry in sorted(master.items()):
        source = bm25_source_text(entry)
        if entry.domain in docs_by_domain:
            docs_by_domain[entry.domain].append((doc_id, source))
        all_docs.append((doc_id, source))

    for domain in DOMAINS:
        pairs = docs_by_domain[domain]
        if not pairs:
            continue
        ids, texts = zip(*pairs)
        index, doc_ids = build_bm25(list(texts), list(ids))
        save_bm25(index, doc_ids, str(out_dir / f"{domain}_bm25.pkl"))

    if all_docs:
        ids, texts = zip(*all_docs)
        index, doc_ids = build_bm25(list(texts), list(ids))
        save_bm25(index, doc_ids, str(out_dir / "all_bm25.pkl"))
