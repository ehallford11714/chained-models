"""Hierarchical PageIndex: document summaries, then heading pages."""

from __future__ import annotations

import math
import re
from collections import Counter

from .corpus import DOCS, Page

_TOKEN = re.compile(r"[a-z0-9_]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall((text or "").lower())


def _tfidf(query: list[str], docs: list[list[str]]) -> list[float]:
    df: Counter[str] = Counter()
    for doc in docs:
        df.update(set(doc))
    n = max(len(docs), 1)
    idf = {term: math.log((n + 1) / (df[term] + 1)) + 1.0 for term in set(query)}
    scores = []
    q_counts = Counter(query)
    for doc in docs:
        counts = Counter(doc)
        score = 0.0
        for term, qf in q_counts.items():
            if not counts[term]:
                continue
            tf = 1.0 + math.log(counts[term])
            score += qf * tf * idf.get(term, 1.0)
        scores.append(score)
    return scores


class PageIndex:
    def __init__(self, pages: tuple[Page, ...] = DOCS):
        self.pages = list(pages)
        self._docs = sorted({page.doc_id for page in self.pages})

    def _doc_blob(self, doc_id: str) -> str:
        parts = [doc_id]
        for page in self.pages:
            if page.doc_id == doc_id:
                parts.append(page.heading)
                parts.append(page.text if page.gold else page.heading)
        return " ".join(parts)

    def retrieve_flat(self, query: str, k: int = 3) -> list[Page]:
        q = tokenize(query)
        bags = [tokenize(f"{page.heading} {page.text}") for page in self.pages]
        ranked = sorted(zip(self.pages, _tfidf(q, bags)), key=lambda row: row[1], reverse=True)
        return [page for page, score in ranked[:k] if score > 0] or [ranked[0][0]]

    def retrieve(self, query: str, k: int = 3) -> list[Page]:
        q = tokenize(query)
        doc_bags = [tokenize(self._doc_blob(doc_id)) for doc_id in self._docs]
        doc_scores = _tfidf(q, doc_bags)
        top_docs = [doc for doc, score in sorted(zip(self._docs, doc_scores), key=lambda row: row[1], reverse=True)[:2]]
        pool = [page for page in self.pages if page.doc_id in top_docs]
        bags = [tokenize(f"{page.doc_id} {page.heading} {page.text}") for page in pool]
        ranked = sorted(zip(pool, _tfidf(q, bags)), key=lambda row: row[1], reverse=True)
        picked = [page for page, score in ranked[:k] if score > 0]
        return picked or pool[:k]
