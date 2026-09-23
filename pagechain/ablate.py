"""Six-way ablation: retrieval × chain depth."""

from __future__ import annotations

import json
from pathlib import Path

from .chain import run_case
from .corpus import CASES
from .pageindex import PageIndex
from .router import Router, entry_pairs, hop_pairs

CONDITIONS = (
    ("direct", "none", 1),
    ("flat", "flat", 1),
    ("pageindex", "pageindex", 1),
    ("chain", "none", 2),
    ("chain_flat", "flat", 2),
    ("chain_pageindex", "pageindex", 2),
)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "artifacts" / "ablation.json"


def _score(case, trace) -> dict:
    recalled = any(case.gold_doc in page for page in trace.pages)
    gold_heads = ("Helpers", "Aliases", "Lab note", "Survival", "Instrument")
    heading_hit = any(case.gold_doc in page and any(head in page for head in gold_heads) for page in trace.pages)
    hop_ok = list(trace.routes[: len(case.hops)]) == list(case.hops)
    answers = case.gold_answer.split()
    got = trace.output
    answer_ok = all(token in got for token in answers)
    return {
        "qid": case.qid,
        "recall": bool(recalled),
        "heading_hit": bool(heading_hit),
        "route_ok": hop_ok,
        "answer_ok": answer_ok,
        "hops": trace.routes,
        "output": got,
        "pages": trace.pages,
    }


def run_ablation() -> dict:
    index = PageIndex()
    entry = Router()
    hop = Router()
    entry.fit(entry_pairs())
    hop.fit(hop_pairs())
    report = {"conditions": {}}
    for name, retrieve, hops in CONDITIONS:
        rows = []
        for case in CASES:
            trace = run_case(case, index=index, entry=entry, hop=hop, retrieve=retrieve, max_hops=hops)
            rows.append(_score(case, trace))
        n = len(rows)
        report["conditions"][name] = {
            "n": n,
            "recall": sum(r["recall"] for r in rows) / n,
            "heading_hit": sum(r["heading_hit"] for r in rows) / n,
            "route_acc": sum(r["route_ok"] for r in rows) / n,
            "answer_acc": sum(r["answer_ok"] for r in rows) / n,
            "rows": rows,
        }
    winner = max(
        report["conditions"].items(),
        key=lambda item: (
            item[1]["answer_acc"],
            item[1]["route_acc"],
            item[1]["heading_hit"],
            1 if item[0] == "chain_pageindex" else 0,
        ),
    )
    report["winner"] = winner[0]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    report = run_ablation()
    print(f"{'condition':18} {'recall':>8} {'heading':>8} {'route':>8} {'answer':>8}")
    for name, block in report["conditions"].items():
        print(f"{name:18} {block['recall']:8.3f} {block['heading_hit']:8.3f} {block['route_acc']:8.3f} {block['answer_acc']:8.3f}")
    print("winner", report["winner"])
    print("wrote", OUT)


if __name__ == "__main__":
    main()
