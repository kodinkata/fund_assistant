from __future__ import annotations

import json
from pathlib import Path

import pytest


def block(text: str, y: float, *, size: float = 10.0, bold: bool = False):
    return {
        "text": text,
        "bbox": [0.0, y, 500.0, y + 12.0],
        "font_size": size,
        "bold": bold,
    }


def page(number: int, blocks: list[dict], *, tables=None, charts=None, images=None):
    text = "\n".join(b["text"] for b in blocks)
    return {
        "page": number,
        "text": text,
        "text_blocks": blocks,
        "tables": tables or [],
        "charts": charts or [],
        "images": images or [],
        "text_for_rag": text,
    }


@pytest.fixture
def registry():
    return [
        {
            "canonical_name": "Cedar Global Fund",
            "aliases": ["Cedar Global Fund", "Cedar Global"],
            "evidence": [],
        },
        {
            "canonical_name": "Oak Income Fund",
            "aliases": ["Oak Income Fund", "Oak Income"],
            "evidence": [],
        },
    ]


@pytest.fixture
def multifund_document():
    list_table = {
        "bbox": [0, 100, 500, 180],
        "rows": [
            ["Name", "Page"],
            ["Cedar Global Fund", "2"],
            ["Oak Income Fund", "3"],
        ],
    }
    directors_table = {
        "bbox": [0, 220, 500, 300],
        "rows": [
            ["Name", "Role"],
            ["Alice Example", "Director"],
            ["Bob Example", "Director"],
        ],
    }

    return {
        "document": "prospectus.pdf",
        "pages": [
            page(
                1,
                [
                    block("List of sub-funds", 20, size=16, bold=True),
                    block("General information applying to all sub-funds.", 60),
                ],
                tables=[list_table, directors_table],
            ),
            page(
                2,
                [
                    block("Cedar Global Fund", 20, size=16, bold=True),
                    block("Investment objective", 50, size=13, bold=True),
                    block("Cedar invests globally in diversified assets.", 80),
                    block("Risk profile", 110, size=13, bold=True),
                    block("The fund may experience market risk.", 140),
                ],
            ),
            page(
                3,
                [
                    block("Oak Income Fund", 20, size=16, bold=True),
                    block("Investment objective", 50, size=13, bold=True),
                    block("Oak seeks regular income from bonds.", 80),
                ],
            ),
        ],
    }


@pytest.fixture
def rights_document():
    return {
        "document": "investor_rights.pdf",
        "pages": [
            page(
                1,
                [
                    block("Transfer of units", 20, size=15, bold=True),
                    block(
                        "Fund units may be transferred freely to another person, "
                        "subject to applicable law.",
                        55,
                    ),
                    block(
                        "Transfers are registered through the applicable securities "
                        "settlement process.",
                        85,
                    ),
                ],
            )
        ],
    }


@pytest.fixture
def single_fund_document():
    return {
        "document": "cedar_kid.pdf",
        "pages": [
            page(
                1,
                [
                    block("Product Name: Cedar Global Fund", 20, size=14, bold=True),
                    block("Investment objective", 55, size=12, bold=True),
                    block("The fund seeks long-term capital growth.", 85),
                ],
            )
        ],
    }


class SimpleBM25:
    """Tiny deterministic lexical scorer used to test retrieval orchestration."""

    def __init__(self, tokenized_corpus: list[list[str]]):
        self.corpus = [set(tokens) for tokens in tokenized_corpus]

    def get_scores(self, query_tokens: list[str]):
        query = set(query_tokens)
        return [float(len(query & doc)) for doc in self.corpus]


@pytest.fixture
def simple_bm25_cls():
    return SimpleBM25
