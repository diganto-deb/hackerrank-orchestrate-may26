from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from exceptions import JSONParseError, ProviderError
from schemas import RetrievedDoc, SubQuery, SubQueryClassification, SubQueryResult
from stages.stage3_retrieve import retrieve
from utils.llm import LLMClient
from utils.prompt_registry import PromptRegistry
from utils.provider import AllProvidersFailedError
from utils.stage_handler import stage_handler

_REGISTRY = PromptRegistry()


def classify_sub_query(
    sub_query: SubQuery,
    retrieved_docs: list[RetrievedDoc],
    context_frame: str,
    llm: LLMClient,
) -> SubQueryResult:
    if not retrieved_docs:
        return SubQueryResult(
            sub_query_id=sub_query.id,
            query=sub_query.query,
            classification=SubQueryClassification.ESCALATION,
            retrieved_docs=[],
        )

    templates, _ = _REGISTRY.get("stage3_classify")
    docs_text = "\n\n".join(
        f"[{d.doc_id}] ({d.product_area})\nTitle: {d.title}\n{d.body_clean[:800]}"
        for d in retrieved_docs
    )
    prompt = templates["template"].format(
        query=sub_query.query, context=context_frame, docs=docs_text
    )
    raw = llm.call_r1([{"role": "user", "content": prompt}])
    data = json.loads(raw)

    return SubQueryResult(
        sub_query_id=sub_query.id,
        query=sub_query.query,
        classification=SubQueryClassification(data["classification"]),
        retrieved_docs=retrieved_docs,
    )


def process_sub_query(
    sub_query: SubQuery,
    bm25_index: Any,
    bm25_ids: list[str],
    master_index: dict,
    request_type: Any,
    context_frame: str,
    llm: LLMClient,
) -> SubQueryResult:
    docs = retrieve(sub_query, bm25_index, bm25_ids, master_index, request_type)
    return classify_sub_query(sub_query, docs, context_frame, llm)


@stage_handler("stage3_classify", maps={json.JSONDecodeError: JSONParseError, AllProvidersFailedError: ProviderError})
def process_sub_queries_parallel(
    sub_queries: list[SubQuery],
    bm25_index: Any,
    bm25_ids: list[str],
    master_index: dict,
    request_type: Any,
    context_frame: str,
    llm: LLMClient,
) -> list[SubQueryResult]:
    if not sub_queries:
        return []

    max_workers = min(len(sub_queries), 4)
    ordered: dict[str, SubQueryResult] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                process_sub_query,
                sub_query,
                bm25_index,
                bm25_ids,
                master_index,
                request_type,
                context_frame,
                llm,
            ): sub_query.id
            for sub_query in sub_queries
        }
        for future in as_completed(futures):
            sub_query_id = futures[future]
            ordered[sub_query_id] = future.result()

    return [ordered[sub_query.id] for sub_query in sub_queries]
