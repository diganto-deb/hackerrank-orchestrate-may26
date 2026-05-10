from __future__ import annotations

import csv
import json
import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

from corpus.enrich import enrich_doc, load_enriched, save_enriched, validate_enrichment
from corpus.index import DOMAINS, build_bm25_indexes, build_master_index, load_master_index, save_master_index
from corpus.manifest import load_manifest, mark_step_complete, save_manifest
from corpus.normalize import normalize_file
from corpus.reconcile import load_unified_taxonomy, reconcile_taxonomies, save_unified_taxonomy
from corpus.taxonomy import build_taxonomy, load_taxonomy, save_taxonomy, validate_taxonomy
from schemas import DomainTaxonomy, EnrichedDoc, NormalizedDoc, UnifiedTaxonomy
from utils.bm25 import load_bm25
from utils.llm import LLMClient, LLMUsage

PROJECT_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_DIR.parent
DATA_DIR = REPO_ROOT / "data"
CACHE_DIR = PROJECT_DIR / "corpus_cache"
NORMALIZED_DIR = CACHE_DIR / "normalized"
TAXONOMY_DIR = CACHE_DIR / "taxonomy"
ENRICHED_DIR = CACHE_DIR / "enriched"
INDEXES_DIR = CACHE_DIR / "indexes"
SAMPLE_TICKETS_PATH = REPO_ROOT / "support_tickets" / "sample_support_tickets.csv"

_COMPANY_TO_DOMAIN = {
    "hackerrank": "hackerrank",
    "claude": "claude",
    "visa": "visa",
}

logger = logging.getLogger(__name__)


def _load_required_labels() -> dict[str, list[str]]:
    """Read sample_support_tickets.csv and extract required product area labels per domain."""
    if not SAMPLE_TICKETS_PATH.exists():
        return {}
    required: dict[str, set[str]] = defaultdict(set)
    with SAMPLE_TICKETS_PATH.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            area = row.get("Product Area", "").strip()
            company = row.get("Company", "").strip().lower()
            domain = _COMPANY_TO_DOMAIN.get(company)
            if area and domain:
                required[domain].add(area)
    return {domain: sorted(labels) for domain, labels in required.items()}


def _load_normalized_from_disk() -> dict[str, list[NormalizedDoc]]:
    all_normalized: dict[str, list[NormalizedDoc]] = {}
    for domain in DOMAINS:
        path = NORMALIZED_DIR / f"{domain}.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        all_normalized[domain] = [NormalizedDoc.model_validate(item) for item in data]
    return all_normalized


def _normalized_artifacts_ready() -> bool:
    try:
        for domain in DOMAINS:
            path = NORMALIZED_DIR / f"{domain}.json"
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, list) or not data:
                return False
        return True
    except Exception:
        return False


def _load_taxonomies_from_disk() -> dict[str, DomainTaxonomy]:
    taxonomies: dict[str, DomainTaxonomy] = {}
    for domain in DOMAINS:
        path = TAXONOMY_DIR / f"{domain}_taxonomy.json"
        if path.exists():
            taxonomies[domain] = load_taxonomy(domain, str(TAXONOMY_DIR))
    return taxonomies


def _taxonomy_artifacts_ready() -> bool:
    try:
        for domain in DOMAINS:
            path = TAXONOMY_DIR / f"{domain}_taxonomy.json"
            taxonomy = load_taxonomy(domain, str(TAXONOMY_DIR))
            if not path.exists() or not taxonomy.labels:
                return False
        return True
    except Exception:
        return False


def _load_enriched_from_disk() -> dict[str, list[EnrichedDoc]]:
    all_enriched: dict[str, list[EnrichedDoc]] = {}
    for domain in DOMAINS:
        path = ENRICHED_DIR / f"{domain}_enriched.json"
        if path.exists():
            all_enriched[domain] = load_enriched(domain, str(ENRICHED_DIR))
    return all_enriched


def _enriched_artifacts_ready() -> bool:
    try:
        for domain in DOMAINS:
            path = ENRICHED_DIR / f"{domain}_enriched.json"
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, list) or not data:
                return False
        return True
    except Exception:
        return False


def _load_master_index_from_disk() -> dict:
    path = ENRICHED_DIR / "master_index.json"
    if not path.exists():
        return {}
    return load_master_index(str(ENRICHED_DIR))


def _unified_taxonomy_ready() -> bool:
    try:
        path = TAXONOMY_DIR / "unified_taxonomy.json"
        if not path.exists():
            return False
        unified = load_unified_taxonomy(str(TAXONOMY_DIR))
        return len(unified.labels) > 0
    except Exception:
        return False


def _master_index_ready() -> bool:
    try:
        path = ENRICHED_DIR / "master_index.json"
        if not path.exists():
            return False
        master = load_master_index(str(ENRICHED_DIR))
        return len(master) > 0
    except Exception:
        return False


def _bm25_indexes_ready() -> bool:
    try:
        for domain in (*DOMAINS, "all"):
            path = INDEXES_DIR / f"{domain}_bm25.pkl"
            if not path.exists() or path.stat().st_size == 0:
                return False
            load_bm25(str(path))
        return True
    except Exception:
        return False


def _make_usage(manifest: dict) -> LLMUsage:
    if hasattr(LLMUsage, "from_dict"):
        try:
            return LLMUsage.from_dict(manifest.get("llm_config", {}))
        except TypeError:
            pass
    return LLMUsage()


def _normalize_domain(domain: str) -> list:
    domain_dir = DATA_DIR / domain
    logger.info("Normalizing domain=%s source_dir=%s", domain, domain_dir)
    docs = []
    for index, path in enumerate(sorted(domain_dir.rglob("*.md")), start=1):
        docs.append(normalize_file(str(path), domain, index))
    logger.info("Normalized domain=%s docs=%d", domain, len(docs))
    return docs


def _step1_normalize(manifest: dict) -> dict[str, list]:
    logger.info("Step 1 start: normalize corpus into %s", NORMALIZED_DIR)
    NORMALIZED_DIR.mkdir(parents=True, exist_ok=True)
    all_normalized: dict[str, list] = {}
    corpus_status = {"total_docs": 0, "total_completed": 0, "domains": {}}

    for domain in DOMAINS:
        docs = _normalize_domain(domain)
        all_normalized[domain] = docs

        out_path = NORMALIZED_DIR / f"{domain}.json"
        out_path.write_text(
            json.dumps([doc.model_dump() for doc in docs], indent=2),
            encoding="utf-8",
        )
        logger.info("Step 1 saved domain=%s normalized_path=%s", domain, out_path)

        with_frontmatter = sum(1 for doc in docs if doc.has_frontmatter)
        with_breadcrumbs = sum(
            1 for doc in docs if doc.existing_metadata.get("breadcrumbs")
        )
        corpus_status["domains"][domain] = {
            "doc_count": len(docs),
            "completed": len(docs),
            "malformed_docs": 0,
            "with_frontmatter": with_frontmatter,
            "with_breadcrumbs": with_breadcrumbs,
        }
        corpus_status["total_docs"] += len(docs)
        corpus_status["total_completed"] += len(docs)

    (NORMALIZED_DIR / "corpus_stats.json").write_text(
        json.dumps(corpus_status, indent=2),
        encoding="utf-8",
    )
    manifest["corpus_status"] = corpus_status
    logger.info(
        "Step 1 complete: total_docs=%d total_completed=%d",
        corpus_status["total_docs"],
        corpus_status["total_completed"],
    )
    return all_normalized


def _step2_taxonomy(all_normalized: dict[str, list], llm: LLMClient, manifest: dict) -> dict:
    logger.info("Step 2 start: taxonomy build into %s", TAXONOMY_DIR)
    TAXONOMY_DIR.mkdir(parents=True, exist_ok=True)
    taxonomies: dict = {}
    taxonomy_stats: dict = {}
    required_labels_by_domain = _load_required_labels()
    logger.info("Step 2 required labels: %s", required_labels_by_domain)

    for domain in DOMAINS:
        docs = all_normalized.get(domain, [])
        if not docs:
            continue

        required_labels = required_labels_by_domain.get(domain)
        logger.info("Step 2 domain=%s docs=%d required_labels=%s", domain, len(docs), required_labels)
        taxonomy = build_taxonomy(domain, docs, llm, required_labels=required_labels)
        gate = validate_taxonomy(taxonomy, docs, required_labels=required_labels)
        manifest["quality_gates"][f"step2_taxonomy_{domain}"] = gate
        logger.info("Step 2 domain=%s gate_passed=%s details=%s", domain, gate["passed"], gate["details"])
        if not gate["passed"]:
            save_manifest(manifest)
            raise SystemExit(f"Taxonomy quality gate failed for {domain}")

        save_taxonomy(taxonomy, str(TAXONOMY_DIR))
        logger.info("Step 2 domain=%s saved taxonomy=%s", domain, TAXONOMY_DIR / f"{domain}_taxonomy.json")
        taxonomies[domain] = taxonomy
        taxonomy_stats[domain] = {
            "label_count": len(taxonomy.labels),
            "labels": [item.product_area_label for item in taxonomy.labels],
            "product_area_doc_counts": {
                item.product_area_label: item.approximate_doc_count
                for item in taxonomy.labels
            },
        }

    manifest["taxonomy_stats"] = taxonomy_stats
    logger.info("Step 2 complete: domains=%d", len(taxonomies))
    return taxonomies


def _step3_enrich(all_normalized: dict[str, list], taxonomies: dict, llm: LLMClient, manifest: dict) -> dict:
    logger.info("Step 3 start: enrichment build into %s", ENRICHED_DIR)
    ENRICHED_DIR.mkdir(parents=True, exist_ok=True)
    all_enriched: dict[str, list] = {}

    quality_gate = {
        "passed": True,
        "low_confidence_count": 0,
        "invented_labels": 0,
        "empty_keywords": 0,
        "flagged_docs": [],
    }

    for domain in DOMAINS:
        docs = all_normalized.get(domain, [])
        taxonomy = taxonomies.get(domain)
        if not docs or taxonomy is None:
            continue

        logger.info("Step 3 domain=%s docs=%d labels=%d", domain, len(docs), len(taxonomy.labels))
        valid_labels = {item.product_area_label for item in taxonomy.labels}
        enriched_docs = []

        for doc in docs:
            enriched = enrich_doc(doc, taxonomy, llm)
            issue = validate_enrichment(enriched, taxonomy, valid_labels)
            if issue:
                quality_gate["flagged_docs"].append(
                    {"doc_id": enriched.doc_id, "reason": issue["reason"]}
                )
                quality_gate["passed"] = False
                if issue.get("hard_fail"):
                    quality_gate["invented_labels"] += 1
                if "confidence" in issue["reason"]:
                    quality_gate["low_confidence_count"] += 1
                if "keywords" in issue["reason"]:
                    quality_gate["empty_keywords"] += 1
                manifest["quality_gates"]["step3_enrichment"] = quality_gate
                save_manifest(manifest)
                logger.error(
                    "Step 3 gate failed domain=%s doc_id=%s reason=%s",
                    domain,
                    enriched.doc_id,
                    issue["reason"],
                )
                raise SystemExit(f"Enrichment quality gate failed for {enriched.doc_id}")

            enriched_docs.append(enriched)

        save_enriched(enriched_docs, domain, str(ENRICHED_DIR))
        logger.info("Step 3 domain=%s saved enriched_path=%s", domain, ENRICHED_DIR / f"{domain}_enriched.json")
        all_enriched[domain] = enriched_docs

    manifest["quality_gates"]["step3_enrichment"] = quality_gate
    logger.info("Step 3 complete: passed=%s flagged_docs=%d", quality_gate["passed"], len(quality_gate["flagged_docs"]))
    if not quality_gate["passed"]:
        save_manifest(manifest)
        raise SystemExit("Enrichment quality gate failed")

    return all_enriched


def _step4_reconcile(taxonomies: dict, llm: LLMClient, manifest: dict):
    logger.info("Step 4 start: reconcile %d domain taxonomies", len(taxonomies))
    unified = reconcile_taxonomies(taxonomies, llm)
    save_unified_taxonomy(unified, str(TAXONOMY_DIR))
    manifest.setdefault("taxonomy_stats", {})["unified_taxonomy"] = {
        "label_count": len(unified.labels),
        "labels": [item.prefixed_label for item in unified.labels],
    }
    logger.info("Step 4 complete: unified_labels=%d saved=%s", len(unified.labels), TAXONOMY_DIR / "unified_taxonomy.json")
    return unified


def _load_or_build_step1(manifest: dict) -> tuple[dict[str, list[NormalizedDoc]], bool]:
    if _normalized_artifacts_ready():
        logger.info("Step 1 already complete; loading normalized artifacts from disk")
        return _load_normalized_from_disk(), True
    logger.info("Step 1 artifacts missing or invalid; rebuilding")
    return _step1_normalize(manifest), False


def _load_or_build_step2(
    all_normalized: dict[str, list[NormalizedDoc]],
    llm: LLMClient,
    manifest: dict,
) -> tuple[dict[str, DomainTaxonomy], bool]:
    if _taxonomy_artifacts_ready():
        logger.info("Step 2 already complete; loading taxonomies from disk")
        return _load_taxonomies_from_disk(), True
    logger.info("Step 2 artifacts missing or invalid; rebuilding")
    return _step2_taxonomy(all_normalized, llm, manifest), False


def _load_or_build_step3(
    all_normalized: dict[str, list[NormalizedDoc]],
    taxonomies: dict[str, DomainTaxonomy],
    llm: LLMClient,
    manifest: dict,
) -> tuple[dict[str, list[EnrichedDoc]], bool]:
    if _enriched_artifacts_ready():
        logger.info("Step 3 already complete; loading enriched docs from disk")
        return _load_enriched_from_disk(), True
    logger.info("Step 3 artifacts missing or invalid; rebuilding")
    return _step3_enrich(all_normalized, taxonomies, llm, manifest), False


def _load_or_build_step4(
    taxonomies: dict[str, DomainTaxonomy],
    llm: LLMClient,
    manifest: dict,
) -> tuple[UnifiedTaxonomy, bool]:
    if _unified_taxonomy_ready():
        if manifest.get("steps_completed", {}).get("step4_reconcile"):
            logger.info("Step 4 already complete; loading unified taxonomy from disk")
        else:
            logger.info("Step 4 artifacts present; loading unified taxonomy from disk")
        return load_unified_taxonomy(str(TAXONOMY_DIR)), True
    logger.info("Step 4 artifacts missing or invalid; rebuilding")
    return _step4_reconcile(taxonomies, llm, manifest), False


def _load_or_build_step5(
    all_normalized: dict[str, list[NormalizedDoc]],
    all_enriched: dict[str, list[EnrichedDoc]],
    unified: UnifiedTaxonomy,
    manifest: dict,
) -> tuple[dict, bool]:
    if _master_index_ready():
        if manifest.get("steps_completed", {}).get("step5_master_index"):
            logger.info("Step 5 already complete; loading master index from disk")
        else:
            logger.info("Step 5 artifacts present; loading master index from disk")
        return _load_master_index_from_disk(), True
    logger.info("Step 5 artifacts missing or invalid; rebuilding")
    return _step5_master_index(all_normalized, all_enriched, unified, manifest), False


def _load_or_build_step6(master: dict, manifest: dict) -> bool:
    if _bm25_indexes_ready():
        logger.info("Step 6 already complete; BM25 indexes already on disk")
        return True
    logger.info("Step 6 artifacts missing or invalid; rebuilding")
    _step6_bm25(master, manifest)
    return False


def _step5_master_index(all_normalized: dict[str, list], all_enriched: dict[str, list], unified, manifest: dict):
    logger.info("Step 5 start: master index build")
    master = build_master_index(all_normalized, all_enriched, unified)
    save_master_index(master, str(ENRICHED_DIR))
    manifest["indexes"]["master_index"] = str(ENRICHED_DIR / "master_index.json")
    logger.info("Step 5 complete: master_docs=%d saved=%s", len(master), ENRICHED_DIR / "master_index.json")
    return master


def _step6_bm25(master: dict, manifest: dict) -> None:
    logger.info("Step 6 start: BM25 build into %s", INDEXES_DIR)
    INDEXES_DIR.mkdir(parents=True, exist_ok=True)
    build_bm25_indexes(master, str(INDEXES_DIR))
    for domain in DOMAINS:
        manifest["indexes"][f"{domain}_bm25"] = str(INDEXES_DIR / f"{domain}_bm25.pkl")
    manifest["indexes"]["all_bm25"] = str(INDEXES_DIR / "all_bm25.pkl")
    logger.info("Step 6 complete: indexes=%s", ", ".join(sorted(manifest["indexes"].keys())))


def _configure_logging() -> None:
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def build_corpus() -> dict:
    logger.info("build_corpus invoked cache_dir=%s data_dir=%s", CACHE_DIR, DATA_DIR)
    manifest = load_manifest()
    usage = _make_usage(manifest)
    llm = LLMClient(usage=usage)
    logger.info("Loaded manifest version=%s steps=%s", manifest.get("version"), manifest.get("steps_completed"))
    save_manifest(manifest)

    upstream_reused = True

    step_start = time.perf_counter()
    all_normalized, step_reused = _load_or_build_step1(manifest)
    if not step_reused:
        mark_step_complete("step1_normalize", manifest, time.perf_counter() - step_start)
        logger.info("Marked step1 complete")
    else:
        logger.info("Skipped step1; using cached normalized corpus")
    upstream_reused = upstream_reused and step_reused

    step_start = time.perf_counter()
    if upstream_reused:
        taxonomies, step_reused = _load_or_build_step2(all_normalized, llm, manifest)
    else:
        logger.info("Step 2 forced to rebuild because an upstream step changed")
        taxonomies = _step2_taxonomy(all_normalized, llm, manifest)
        step_reused = False
    manifest["llm_config"] = usage.to_dict()
    if not step_reused:
        mark_step_complete("step2_taxonomy", manifest, time.perf_counter() - step_start)
        logger.info("Marked step2 complete")
    else:
        logger.info("Skipped step2; using cached taxonomies")
    upstream_reused = upstream_reused and step_reused

    step_start = time.perf_counter()
    if upstream_reused:
        all_enriched, step_reused = _load_or_build_step3(all_normalized, taxonomies, llm, manifest)
    else:
        logger.info("Step 3 forced to rebuild because an upstream step changed")
        all_enriched = _step3_enrich(all_normalized, taxonomies, llm, manifest)
        step_reused = False
    manifest["llm_config"] = usage.to_dict()
    if not step_reused:
        mark_step_complete("step3_enrich", manifest, time.perf_counter() - step_start)
        logger.info("Marked step3 complete")
    else:
        logger.info("Skipped step3; using cached enrichments")
    upstream_reused = upstream_reused and step_reused

    step_start = time.perf_counter()
    if upstream_reused:
        unified, step_reused = _load_or_build_step4(taxonomies, llm, manifest)
    else:
        logger.info("Step 4 forced to rebuild because an upstream step changed")
        unified = _step4_reconcile(taxonomies, llm, manifest)
        step_reused = False
    manifest["llm_config"] = usage.to_dict()
    if not step_reused:
        mark_step_complete("step4_reconcile", manifest, time.perf_counter() - step_start)
        logger.info("Marked step4 complete")
    else:
        logger.info("Skipped step4; using cached unified taxonomy")
    upstream_reused = upstream_reused and step_reused

    step_start = time.perf_counter()
    if upstream_reused:
        master, step_reused = _load_or_build_step5(all_normalized, all_enriched, unified, manifest)
    else:
        logger.info("Step 5 forced to rebuild because an upstream step changed")
        master = _step5_master_index(all_normalized, all_enriched, unified, manifest)
        step_reused = False
    if not step_reused:
        mark_step_complete("step5_master_index", manifest, time.perf_counter() - step_start)
        logger.info("Marked step5 complete")
    else:
        logger.info("Skipped step5; using cached master index")
    upstream_reused = upstream_reused and step_reused

    step_start = time.perf_counter()
    if upstream_reused:
        step_reused = _load_or_build_step6(master, manifest)
    else:
        logger.info("Step 6 forced to rebuild because an upstream step changed")
        _step6_bm25(master, manifest)
        step_reused = False
    manifest["llm_config"] = usage.to_dict()
    if not step_reused:
        manifest["build_end"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        mark_step_complete("step6_bm25", manifest, time.perf_counter() - step_start)
        logger.info("Marked step6 complete")
    else:
        logger.info("Skipped step6; BM25 indexes already cached")
    return manifest


def main() -> None:
    _configure_logging()
    build_corpus()


if __name__ == "__main__":
    try:
        _configure_logging()
        main()
    except SystemExit:
        raise
    except Exception as exc:  # pragma: no cover
        logger.exception("build_corpus failed")
        print(f"build_corpus failed: {exc}", file=sys.stderr)
        raise
