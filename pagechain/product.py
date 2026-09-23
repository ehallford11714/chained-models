"""Ask PageChain a question with the winning condition: chain + PageIndex."""

from __future__ import annotations

from .chain import run_case
from .corpus import Case
from .pageindex import PageIndex
from .router import Router, entry_pairs, hop_pairs


def _bank():
    entry = Router()
    hop = Router()
    entry.fit(entry_pairs())
    hop.fit(hop_pairs())
    return PageIndex(), entry, hop


def ask(query: str) -> dict:
    index, entry, hop = _bank()
    case = Case("ask", query, "code", "", "", ("code",))
    trace = run_case(case, index=index, entry=entry, hop=hop, retrieve="pageindex", max_hops=2)
    return {"query": query, "hops": trace.routes, "pages": trace.pages, "output": trace.output}


def main() -> None:
    import sys

    query = " ".join(sys.argv[1:]) or "What PAM does the lab note use?"
    result = ask(query)
    print(result["output"] or "(no fact in retrieved pages)")
    print("hops", " → ".join(result["hops"]))
    print("pages", ", ".join(result["pages"]))


if __name__ == "__main__":
    main()
