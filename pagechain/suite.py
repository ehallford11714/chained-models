"""Chained Models suite: index, train, ask, ablate, save."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from .ablate import run_ablation
from .chain import run_case
from .corpus import CASES, Case, DOCS
from .ingest import index_dir
from .pageindex import PageIndex
from .router import Router, entry_pairs, hop_pairs

ROOT = Path(__file__).resolve().parent.parent
CKPT = ROOT / "checkpoints"


class ChainedModels:
    def __init__(self, index: PageIndex | None = None):
        self.index = index or PageIndex(DOCS)
        self.entry = Router()
        self.hop = Router()

    def index_markdown(self, folder: Path) -> int:
        self.index = index_dir(folder)
        return len(self.index.pages)

    def train(self) -> dict[str, float]:
        return {
            "entry_loss": self.entry.fit(entry_pairs()),
            "hop_loss": self.hop.fit(hop_pairs()),
        }

    def save(self, directory: Path | None = None) -> Path:
        directory = directory or CKPT
        directory.mkdir(parents=True, exist_ok=True)
        torch.save({"state": self.entry.model.state_dict(), "ids": self.entry.ids}, directory / "entry.pt")
        torch.save({"state": self.hop.model.state_dict(), "ids": self.hop.ids}, directory / "hop.pt")
        return directory

    def load(self, directory: Path | None = None) -> None:
        directory = directory or CKPT
        for router, name in ((self.entry, "entry.pt"), (self.hop, "hop.pt")):
            blob = torch.load(directory / name, map_location="cpu", weights_only=False)
            router.model.load_state_dict(blob["state"])
            router.ids = list(blob["ids"])

    def ask(self, query: str, *, hops: int = 2, retrieve: str = "pageindex") -> dict:
        case = Case("ask", query, "code", "", "", ("code",))
        trace = run_case(
            case,
            index=self.index,
            entry=self.entry,
            hop=self.hop,
            retrieve=retrieve,
            max_hops=hops,
        )
        return {
            "query": query,
            "hops": trace.routes,
            "pages": trace.pages,
            "output": trace.output,
        }

    def ablate(self) -> dict:
        report = run_ablation()
        return {name: {k: v for k, v in block.items() if k != "rows"} for name, block in report["conditions"].items()} | {
            "winner": report["winner"]
        }

    def eval_cases(self) -> list[dict]:
        rows = []
        for case in CASES:
            trace = run_case(
                case,
                index=self.index,
                entry=self.entry,
                hop=self.hop,
                retrieve="pageindex",
                max_hops=max(2, len(case.hops)),
            )
            rows.append({"qid": case.qid, "hops": trace.routes, "output": trace.output})
        return rows


def write_report(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
