"""Planted multi-domain corpus. One gold heading per document."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Page:
    doc_id: str
    heading: str
    domain: str
    text: str
    gold: bool


@dataclass(frozen=True)
class Case:
    qid: str
    query: str
    gold_domain: str
    gold_answer: str
    gold_doc: str
    hops: tuple[str, ...]


DOCS: tuple[Page, ...] = (
    Page("code_pytest", "Setup", "code", "Install pytest and pin the plugin wheel. Ignore the helper name here.", False),
    Page("code_pytest", "Helpers", "code", "The test helper is named assert_route. Use it in mote routing tests.", True),
    Page("code_pytest", "CI", "code", "The workflow file is named weekly.yml and does not mention helpers.", False),
    Page("sql_revenue", "Warehouse", "sql", "Raw landings sit in bronze.rev_raw. Do not query that for FY reports.", False),
    Page("sql_revenue", "Aliases", "sql", "Revenue table alias is rev_fy26. Join it to dim_account on account_id.", True),
    Page("sql_revenue", "Access", "sql", "Analysts request a viewer role. No alias is granted in this section.", False),
    Page("science_crispr", "Background", "science", "Cas9 is an RNA-guided nuclease. This section omits the lab PAM.", False),
    Page("science_crispr", "Lab note", "science", "PAM sequence used in the lab note is NGG. Spacer length is 20 nt.", True),
    Page("science_crispr", "Safety", "science", "Off-target screens are required. No PAM is restated here.", False),
    Page("law_nda", "Parties", "law", "The discloser is North Pier Labs. Term is not the survival window.", False),
    Page("law_nda", "Survival", "law", "The survival clause lasts 36 months after termination.", True),
    Page("law_nda", "Venue", "law", "Disputes go to New Jersey. Survival is not restated.", False),
    Page("finance_iv", "Returns", "finance", "The outcome Y is excess return. Do not treat the year dummy as Y.", False),
    Page("finance_iv", "Instrument", "finance", "The instrument Z is branch_open_year. First-stage F must exceed 10.", True),
    Page("finance_iv", "Placebo", "finance", "A lag of peer returns is the placebo. It is not the instrument.", False),
    Page(
        "decoy_faq",
        "FAQ",
        "code",
        "People ask the pytest helper name, the revenue alias, the PAM, the survival clause, "
        "and the instrument Z. This FAQ records none of those values.",
        False,
    ),
)

CASES: tuple[Case, ...] = (
    Case("c1", "What is the pytest helper name?", "code", "assert_route", "code_pytest", ("code",)),
    Case("c2", "What alias is used for the revenue table?", "sql", "rev_fy26", "sql_revenue", ("sql",)),
    Case("c3", "What PAM does the lab note use?", "science", "NGG", "science_crispr", ("science",)),
    Case("c4", "How long does the NDA survival clause last?", "law", "36 months", "law_nda", ("law",)),
    Case("c5", "What is the instrumental variable Z?", "finance", "branch_open_year", "finance_iv", ("finance",)),
    Case(
        "h1",
        "Find the lab PAM and then the revenue table alias.",
        "science",
        "NGG rev_fy26",
        "science_crispr",
        ("science", "sql"),
    ),
    Case(
        "h2",
        "Name the pytest helper and then the NDA survival window.",
        "code",
        "assert_route 36 months",
        "code_pytest",
        ("code", "law"),
    ),
    Case(
        "h3",
        "What instrument Z is used, then which PAM is in the lab note?",
        "finance",
        "branch_open_year NGG",
        "finance_iv",
        ("finance", "science"),
    ),
)

DOMAINS = ("code", "sql", "science", "law", "finance")
