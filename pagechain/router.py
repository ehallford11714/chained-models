"""Hash routers for entry and hops. Same family as MoTE, own labels."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .corpus import CASES, DOMAINS

BUCKETS = 2048
SEQ = 48


def hash_ids(text: str) -> list[int]:
    words = (text or "").lower().replace("/", " ").replace(".", " ").split()
    ids = [1 + (zlib_crc(w) % (BUCKETS - 1)) for w in words[:SEQ]]
    if not ids:
        ids = [0]
    return (ids + [0] * SEQ)[:SEQ]


def zlib_crc(word: str) -> int:
    import zlib

    return zlib.crc32(word.encode("utf-8"))


class Scorer(nn.Module):
    def __init__(self, n_experts: int = len(DOMAINS), dim: int = 64):
        super().__init__()
        self.embed = nn.Embedding(BUCKETS, dim, padding_idx=0)
        self.norm = nn.LayerNorm(dim)
        self.scorer = nn.Linear(dim, n_experts)
        nn.init.normal_(self.embed.weight, std=0.02)
        with torch.no_grad():
            self.embed.weight[0].zero_()

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(token_ids)
        mask = (token_ids != 0).float().unsqueeze(-1)
        pooled = (x * mask).sum(1) / mask.sum(1).clamp_min(1.0)
        return self.scorer(self.norm(pooled))


@dataclass
class Route:
    domain: str
    confidence: float


class Router:
    def __init__(self):
        self.model = Scorer()
        self.ids = list(DOMAINS)

    def route(self, text: str) -> Route:
        ids = torch.tensor([hash_ids(text)], dtype=torch.long)
        self.model.eval()
        with torch.no_grad():
            prob = F.softmax(self.model(ids)[0], dim=-1)
        idx = int(prob.argmax())
        return Route(self.ids[idx], float(prob[idx]))

    def fit(self, pairs: list[tuple[str, str]], steps: int = 80) -> float:
        opt = torch.optim.AdamW(self.model.parameters(), lr=3e-3)
        label = {name: i for i, name in enumerate(self.ids)}
        last = 0.0
        self.model.train()
        for _ in range(steps):
            texts, y = zip(*pairs)
            x = torch.tensor([hash_ids(t) for t in texts], dtype=torch.long)
            targets = torch.tensor([label[name] for name in y], dtype=torch.long)
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(self.model(x), targets)
            loss.backward()
            opt.step()
            last = float(loss.detach())
        return last


def entry_pairs() -> list[tuple[str, str]]:
    rows = [(case.query, case.gold_domain) for case in CASES]
    extras = [
        ("fix the pytest helper", "code"),
        ("write the revenue join alias", "sql"),
        ("crispr lab pam sequence", "science"),
        ("nda survival months", "law"),
        ("instrumental variable branch year", "finance"),
    ]
    return rows + extras


def hop_pairs() -> list[tuple[str, str]]:
    return [
        ("assert_route", "law"),
        ("assert_route next find the survival window", "law"),
        ("NGG", "sql"),
        ("NGG next find the revenue alias", "sql"),
        ("branch_open_year", "science"),
        ("branch_open_year next find the lab PAM", "science"),
        ("rev_fy26", "sql"),
        ("36 months", "law"),
    ]
