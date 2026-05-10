from __future__ import annotations

from dataclasses import dataclass, field

from rank_bm25 import BM25Okapi

from schemas import UnifiedTaxonomy

RELEVANCE_THRESHOLD = 0.1
HIGH_CONFIDENCE_COUNT = 4
TOP_K = 5


@dataclass
class DomainInferenceResult:
    inferred_domain: str | None
    is_cross_domain: bool
    short_circuit: bool
    cross_domain_indexes: list[str] = field(default_factory=list)


def _taxonomy_keyword_match(query: str, taxonomy: UnifiedTaxonomy) -> set[str]:
    query_lower = query.lower()
    matched_domains: set[str] = set()
    for label in taxonomy.labels:
        terms = label.bare_label.replace("-", " ").split() + label.description.lower().split()
        if any(term in query_lower for term in terms if len(term) > 3):
            matched_domains.add(label.domain)
    return matched_domains


def infer_domain(
    reformulated_query: str,
    unified_taxonomy: UnifiedTaxonomy,
    all_indexes: dict[str, tuple[BM25Okapi, list[str]]],
) -> DomainInferenceResult:
    all_index, all_ids = all_indexes["all"]
    tokens = reformulated_query.lower().split()
    scores = all_index.get_scores(tokens)
    ranked = sorted(zip(all_ids, scores), key=lambda item: item[1], reverse=True)
    top_k = [(doc_id, score) for doc_id, score in ranked[:TOP_K] if score > RELEVANCE_THRESHOLD]

    taxonomy_domains = _taxonomy_keyword_match(reformulated_query, unified_taxonomy)
    if not top_k and not taxonomy_domains:
        return DomainInferenceResult(
            inferred_domain=None,
            is_cross_domain=False,
            short_circuit=True,
        )

    domain_counts: dict[str, int] = {}
    for doc_id, _ in top_k:
        domain = doc_id.split("_")[0]
        domain_counts[domain] = domain_counts.get(domain, 0) + 1

    if domain_counts:
        top_domain = max(domain_counts, key=lambda domain_name: domain_counts[domain_name])
        if domain_counts[top_domain] >= HIGH_CONFIDENCE_COUNT:
            return DomainInferenceResult(
                inferred_domain=top_domain,
                is_cross_domain=False,
                short_circuit=False,
            )

    candidate_domains = set(domain_counts.keys()) | taxonomy_domains
    if len(candidate_domains) == 1:
        return DomainInferenceResult(
            inferred_domain=next(iter(candidate_domains)),
            is_cross_domain=False,
            short_circuit=False,
        )
    if len(candidate_domains) > 1:
        return DomainInferenceResult(
            inferred_domain=None,
            is_cross_domain=True,
            short_circuit=False,
            cross_domain_indexes=sorted(candidate_domains),
        )

    return DomainInferenceResult(
        inferred_domain=None,
        is_cross_domain=False,
        short_circuit=True,
    )
