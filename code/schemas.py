from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class DocType(str, Enum):
    FAQ = "faq"
    TROUBLESHOOTING = "troubleshooting"
    POLICY = "policy"
    HOW_TO = "how-to"
    REFERENCE = "reference"


class RequestType(str, Enum):
    PRODUCT_ISSUE = "product_issue"
    BUG = "bug"
    FEATURE_REQUEST = "feature_request"
    INVALID = "invalid"


class Status(str, Enum):
    REPLIED = "replied"
    ESCALATED = "escalated"
    ERROR = "error"


class SubQueryClassification(str, Enum):
    ANSWERABLE = "answerable"
    ESCALATION = "escalation"
    INVALID = "invalid"


class NormalizedDoc(BaseModel):
    doc_id: str
    domain: str
    source_path: str
    title: str
    first_paragraph: str
    body_clean: str
    body_raw: str
    existing_metadata: dict = Field(default_factory=dict)
    has_frontmatter: bool
    frontmatter_schema: Optional[str] = None


class TaxonomyLabel(BaseModel):
    product_area_label: str
    description: str
    example_doc_ids: list[str]
    approximate_doc_count: int


class DomainTaxonomy(BaseModel):
    domain: str
    overview: str
    labels: list[TaxonomyLabel]


class EnrichedDoc(BaseModel):
    doc_id: str
    product_area_primary: str
    doc_type: DocType
    summary: str
    keywords: list[str]
    confidence: float
    cross_refs: list[str] = Field(default_factory=list)


class UnifiedLabel(BaseModel):
    domain: str
    bare_label: str
    prefixed_label: str
    description: str
    doc_count: int


class UnifiedTaxonomy(BaseModel):
    labels: list[UnifiedLabel]

    def label_set(self) -> set[str]:
        return {ul.prefixed_label for ul in self.labels}

    def bare_label_set(self) -> set[str]:
        return {ul.bare_label for ul in self.labels}

    def prefixed_labels_for_domain(self, domain: str) -> list[str]:
        return [ul.prefixed_label for ul in self.labels if ul.domain == domain]


class MasterIndexEntry(BaseModel):
    domain: str
    product_area: str
    doc_type: DocType
    summary: str
    keywords: list[str]
    title: str
    source_url: str = ""
    body_clean: str = ""


class Ticket(BaseModel):
    issue: str
    subject: str = ""
    company: Optional[str] = None


class SubQuery(BaseModel):
    id: str
    query: str
    context_ref: str = "shared_context"


class SharedContext(BaseModel):
    domain: Optional[str] = None
    situational_frame: str
    company: Optional[str] = None


class DecompositionResult(BaseModel):
    shared_context: SharedContext
    sub_queries: list[SubQuery]


class RetrievedDoc(BaseModel):
    doc_id: str
    title: str
    body_clean: str
    product_area: str
    doc_type: DocType
    score: float


class SubQueryResult(BaseModel):
    sub_query_id: str
    query: str
    classification: SubQueryClassification
    retrieved_docs: list[RetrievedDoc]


class OutputRow(BaseModel):
    status: Status
    product_area: str
    response: str
    justification: str
    request_type: RequestType
    prompt_versions: dict[str, str] = Field(default_factory=dict)
