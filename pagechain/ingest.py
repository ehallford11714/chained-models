"""Turn markdown files into a PageIndex. Headings become pages."""

from __future__ import annotations

import re
from pathlib import Path

from .corpus import Page
from .pageindex import PageIndex

_HEAD = re.compile(r"^(#{1,3})\s+(.+)$")


def pages_from_markdown(root: Path, domain: str = "docs") -> list[Page]:
    pages: list[Page] = []
    for path in sorted(root.rglob("*.md")):
        doc_id = path.stem
        heading = path.stem
        buf: list[str] = []
        blocks: list[tuple[str, str]] = []

        def flush() -> None:
            body = "\n".join(buf).strip()
            if body:
                blocks.append((heading, body))

        for line in path.read_text(encoding="utf-8").splitlines():
            match = _HEAD.match(line)
            if match:
                flush()
                heading = match.group(2).strip()
                buf = []
            else:
                buf.append(line)
        flush()
        if not blocks:
            continue
        for title, body in blocks:
            pages.append(Page(doc_id, title, domain, body, False))
    return pages


def index_dir(root: Path, domain: str = "docs") -> PageIndex:
    pages = pages_from_markdown(root, domain=domain)
    if not pages:
        raise FileNotFoundError(f"no markdown under {root}")
    return PageIndex(tuple(pages))
