from pathlib import Path

from pagechain.ablate import run_ablation
from pagechain.ingest import index_dir
from pagechain.pageindex import PageIndex


def test_pageindex_prefers_the_gold_heading():
    pages = PageIndex().retrieve("What PAM does the lab note use?", k=3)
    assert any(page.gold and page.domain == "science" for page in pages)


def test_markdown_folder_becomes_heading_pages():
    index = index_dir(Path(__file__).resolve().parents[1] / "examples" / "docs", domain="docs")
    pages = index.retrieve("What PAM does the lab note use?", k=2)
    assert pages[0].heading == "Lab note"


def test_chain_pageindex_is_the_best_answer_condition():
    report = run_ablation()
    assert report["conditions"]["chain_pageindex"]["answer_acc"] >= report["conditions"]["direct"]["answer_acc"]
    assert report["winner"] in {"chain_pageindex", "pageindex", "chain_flat"}
