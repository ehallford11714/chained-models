"""Chained Models: PageIndex retrieval plus trained hop routers."""

__all__ = ["ChainedModels"]


def __getattr__(name: str):
    if name in {"ChainedModels", "PageChainSuite"}:
        from .suite import ChainedModels

        return ChainedModels
    raise AttributeError(name)
