from __future__ import annotations

from rank_bm25 import BM25Okapi

from schemas import DocType, MasterIndexEntry, RequestType, RetrievedDoc, SubQuery
from utils.bm25 import search_bm25

_DOC_TYPE_RANK: dict[RequestType, list[DocType]] = {
    RequestType.BUG: [
        DocType.TROUBLESHOOTING,
        DocType.REFERENCE,
        DocType.HOW_TO,
        DocType.POLICY,
        DocType.FAQ,
    ],
    RequestType.PRODUCT_ISSUE: [
        DocType.TROUBLESHOOTING,
        DocType.HOW_TO,
        DocType.POLICY,
        DocType.REFERENCE,
        DocType.FAQ,
    ],
    RequestType.FEATURE_REQUEST: [
        DocType.HOW_TO,
        DocType.REFERENCE,
        DocType.TROUBLESHOOTING,
        DocType.POLICY,
        DocType.FAQ,
    ],
    RequestType.INVALID: [
        DocType.FAQ,
        DocType.HOW_TO,
        DocType.TROUBLESHOOTING,
        DocType.POLICY,
        DocType.REFERENCE,
    ],
}

TOP_K = 5
RERANK_BOOST = 0.2


def _doc_type_boost(doc_type: DocType, request_type: RequestType) -> float:
    priority_types = _DOC_TYPE_RANK.get(request_type, [])
    if doc_type not in priority_types:
        return 0.0
    rank = priority_types.index(doc_type)
    return RERANK_BOOST * (len(priority_types) - rank)


def retrieve(
    sub_query: SubQuery,
    bm25_index: BM25Okapi,
    bm25_ids: list[str],
    master_index: dict[str, MasterIndexEntry],
    request_type: RequestType,
    top_k: int = TOP_K,
) -> list[RetrievedDoc]:
    results = search_bm25(bm25_index, bm25_ids, sub_query.query, top_k=top_k * 2)

    scored_docs: list[tuple[float, RetrievedDoc]] = []
    for doc_id, bm25_score in results:
        entry = master_index.get(doc_id)
        if entry is None:
            continue

        final_score = bm25_score + _doc_type_boost(entry.doc_type, request_type)
        scored_docs.append(
            (
                final_score,
                RetrievedDoc(
                    doc_id=doc_id,
                    title=entry.title,
                    body_clean=entry.body_clean[:3000],
                    product_area=entry.product_area,
                    doc_type=entry.doc_type,
                    score=final_score,
                ),
            )
        )

    scored_docs.sort(key=lambda item: item[0], reverse=True)
    return [doc for _, doc in scored_docs[:top_k]]
