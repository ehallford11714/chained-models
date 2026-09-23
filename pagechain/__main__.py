"""PageChain command line.

    python -m pagechain ablate
    python -m pagechain train
    python -m pagechain index examples/docs
    python -m pagechain ask What PAM does the lab note use?
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .suite import PageChainSuite


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="pagechain", description="PageIndex plus trained hop routers")
    sub = parser.add_subparsers(dest="cmd")
    sub.add_parser("ablate", help="Run the six-condition ablation")
    sub.add_parser("train", help="Train entry and hop routers and save checkpoints")
    index = sub.add_parser("index", help="Build a PageIndex from a markdown folder")
    index.add_argument("folder", type=Path)
    ask = sub.add_parser("ask", help="Route a question through PageIndex and up to N hops")
    ask.add_argument("query", nargs="+")
    ask.add_argument("--hops", type=int, default=2)
    ask.add_argument("--retrieve", choices=("pageindex", "flat", "none"), default="pageindex")
    args = parser.parse_args(argv)

    if args.cmd is None:
        parser.print_help()
        return
    suite = PageChainSuite()
    if args.cmd == "ablate":
        report = suite.ablate()
        print(json.dumps(report, indent=2))
        return
    if args.cmd == "index":
        count = suite.index_markdown(args.folder)
        print(f"indexed {count} pages from {args.folder}")
        return
    if args.cmd == "train":
        losses = suite.train()
        path = suite.save()
        print(json.dumps(losses))
        print("saved", path)
        return
    if args.cmd == "ask":
        suite.train()
        result = suite.ask(" ".join(args.query), hops=args.hops, retrieve=args.retrieve)
        print(result["output"] or "(no fact in retrieved pages)")
        print("hops", " -> ".join(result["hops"]))
        print("pages", ", ".join(result["pages"]))


if __name__ == "__main__":
    main()
