"""Closed-book stub specialists. They only speak a fact their pages contain."""

from __future__ import annotations

from .corpus import DOMAINS, Page

FACTS = {
    "code": "assert_route",
    "sql": "rev_fy26",
    "science": "NGG",
    "law": "36 months",
    "finance": "branch_open_year",
}


def answer(domain: str, pages: list[Page], prior: str = "") -> str:
    if domain not in DOMAINS:
        return (prior or "").strip()
    blob = " ".join(page.text for page in pages)
    found = FACTS[domain] if FACTS[domain] in blob else ""
    prior = (prior or "").strip()
    if found and prior and found not in prior:
        return f"{prior} {found}"
    return found or prior
