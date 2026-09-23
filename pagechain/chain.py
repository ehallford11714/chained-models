"""One or two hops. Optional PageIndex or flat retrieve at each hop."""

from __future__ import annotations

from dataclasses import dataclass, field

from .corpus import Case
from .experts import answer
from .pageindex import PageIndex
from .router import Router


@dataclass
class Trace:
    hops: list[str] = field(default_factory=list)
    pages: list[str] = field(default_factory=list)
    output: str = ""
    routes: list[str] = field(default_factory=list)


def run_case(
    case: Case,
    *,
    index: PageIndex,
    entry: Router,
    hop: Router,
    retrieve: str,
    max_hops: int,
) -> Trace:
    trace = Trace()
    note = case.query
    used: set[str] = set()
    for step in range(max_hops):
        router = entry if step == 0 else hop
        decision = router.route(note)
        domain = decision.domain
        if domain in used:
            break
        used.add(domain)
        fetch = case.query if not trace.output else f"{case.query} {trace.output}"
        if retrieve == "pageindex":
            pages = index.retrieve(fetch, k=3)
        elif retrieve == "flat":
            pages = index.retrieve_flat(fetch, k=3)
        else:
            pages = []
        note = answer(domain, pages, prior=trace.output)
        trace.hops.append(domain)
        trace.routes.append(domain)
        trace.pages.extend(f"{page.doc_id}#{page.heading}" for page in pages)
        trace.output = note
        if not note:
            break
    return trace
