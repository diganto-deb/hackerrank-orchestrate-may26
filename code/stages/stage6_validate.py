from __future__ import annotations

import re

from schemas import OutputRow, RequestType, Status

_DOMAIN_LABEL_RE = re.compile(r"^[a-z][a-z0-9-]+:[a-z][a-z0-9-]+$")
_BARE_LABEL_RE = re.compile(r"^[a-z][a-z0-9_-]+$")

_VALID_STATUSES = {Status.REPLIED, Status.ESCALATED}
_VALID_REQUEST_TYPES = {
    RequestType.PRODUCT_ISSUE,
    RequestType.BUG,
    RequestType.FEATURE_REQUEST,
    RequestType.INVALID,
}


def _matches_domain_label_format(label: str) -> bool:
    return bool(_DOMAIN_LABEL_RE.match(label))


def _normalize_product_area(label: str) -> str:
    if ":" in label:
        return label.split(":", 1)[1]
    return label


def _matches_bare_label_format(label: str) -> bool:
    return bool(_BARE_LABEL_RE.match(label))


def validate_output(output: OutputRow, unified_taxonomy_labels: set[str]) -> OutputRow:
    failures: list[str] = []

    if output.status not in _VALID_STATUSES:
        failures.append(f"invalid status: {output.status!r}")

    if output.request_type not in _VALID_REQUEST_TYPES:
        failures.append(f"invalid request_type: {output.request_type!r}")

    if not output.response.strip():
        failures.append("response is empty")

    if not output.justification.strip():
        failures.append("justification is empty")

    normalized_product_area = _normalize_product_area(output.product_area)
    if output.request_type == RequestType.INVALID:
        if output.product_area not in ("", "none") and not _matches_bare_label_format(normalized_product_area):
            failures.append(
                f"product_area format invalid for invalid request: {output.product_area!r}"
            )
    else:
        if not _matches_bare_label_format(normalized_product_area):
            failures.append(f"product_area format invalid: {output.product_area!r}")
        elif normalized_product_area not in unified_taxonomy_labels:
            failures.append(f"product_area not in taxonomy: {output.product_area!r}")

    if not failures:
        return output

    request_type = (
        output.request_type if output.request_type in _VALID_REQUEST_TYPES else RequestType.INVALID
    )
    response = output.response or "Escalated due to schema validation failure."
    return OutputRow(
        status=Status.ESCALATED,
        product_area=normalized_product_area,
        response=response,
        justification=f"Schema validation failed: {'; '.join(failures)}",
        request_type=request_type,
    )
