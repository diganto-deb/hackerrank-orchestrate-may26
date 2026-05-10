from __future__ import annotations

import html
import re
from pathlib import Path

import yaml

from schemas import NormalizedDoc

_FRONTMATTER_DELIMITERS = ("---", "+++")


def parse_frontmatter(text: str) -> tuple[dict, str]:
    for delim in _FRONTMATTER_DELIMITERS:
        if not text.startswith(delim):
            continue

        lines = text.splitlines()
        if not lines or lines[0].strip() != delim:
            continue

        end_index = None
        for index in range(1, len(lines)):
            if lines[index].strip() == delim:
                end_index = index
                break

        if end_index is None:
            return {}, text

        raw_fm = "\n".join(lines[1:end_index])
        body = "\n".join(lines[end_index + 1 :]).lstrip()

        try:
            frontmatter = yaml.safe_load(raw_fm) or {}
            if isinstance(frontmatter, dict):
                return frontmatter, body
            return {}, body
        except yaml.YAMLError:
            title_match = re.search(r'^title:\s*["\']?(.+?)["\']?\s*$', raw_fm, re.MULTILINE)
            if title_match:
                return {"title": title_match.group(1).strip()}, body
            return {}, body

    return {}, text


def strip_html(text: str) -> str:
    text = re.sub(r"<pre>\s*<code[^>]*>(.*?)</code>\s*</pre>", r"```\n\1\n```", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    text = html.unescape(text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_title(frontmatter: dict, body: str, filename: str) -> str:
    if frontmatter.get("title"):
        return str(frontmatter["title"]).strip()

    match = re.search(r"^#\s+(.+)$", body, re.MULTILINE)
    if match:
        return match.group(1).strip()

    stem = Path(filename).stem if "." in filename else filename
    return stem.replace("_", " ").replace("-", " ").title()


def extract_first_paragraph(body: str) -> str:
    lines = body.splitlines()
    past_title = False
    paragraph_lines: list[str] = []

    for line in lines:
        stripped = line.strip()

        if not past_title:
            if stripped.startswith("# "):
                past_title = True
            continue

        if not stripped:
            if paragraph_lines:
                break
            continue

        if stripped.startswith("#"):
            break

        if stripped.startswith(("- ", "* ", "| ", "1.")):
            if not paragraph_lines:
                paragraph_lines.append(stripped)
            break

        paragraph_lines.append(stripped)

    if paragraph_lines:
        return " ".join(paragraph_lines)

    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and not stripped.startswith("---"):
            return stripped

    return ""


def normalize_file(path: str, domain: str, doc_index: int) -> NormalizedDoc:
    doc_id = f"{domain}_{doc_index:04d}"
    raw = Path(path).read_text(encoding="utf-8", errors="replace")
    frontmatter, body_raw = parse_frontmatter(raw)
    body_clean = strip_html(body_raw)
    title = extract_title(frontmatter, body_clean, Path(path).stem)
    first_paragraph = extract_first_paragraph(body_clean)

    existing_metadata = {
        "source_url": frontmatter.get("source_url") or frontmatter.get("url", ""),
        "breadcrumbs": frontmatter.get("breadcrumbs") or [],
        "article_id": frontmatter.get("article_id") or frontmatter.get("article_slug", ""),
    }

    return NormalizedDoc(
        doc_id=doc_id,
        domain=domain,
        source_path=path,
        title=title,
        first_paragraph=first_paragraph,
        body_clean=body_clean,
        body_raw=body_raw,
        existing_metadata=existing_metadata,
        has_frontmatter=bool(frontmatter),
        frontmatter_schema=domain if frontmatter else None,
    )
