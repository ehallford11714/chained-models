"""PageChain: PageIndex retrieval plus trained hop routers."""

__all__ = ["PageChainSuite"]


def __getattr__(name: str):
    if name == "PageChainSuite":
        from .suite import PageChainSuite

        return PageChainSuite
    raise AttributeError(name)
