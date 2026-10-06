from __future__ import annotations

import json

import pytest

from fund_rag.generator import (
    GeminiAnswer,
    GeminiFundAssistant,
    PromptAugmenter,
    validate_answer,
)


def evidence(chunk_id, fund, text, document="doc.pdf", page=1, section="Section"):
    return {
        "chunk_id": chunk_id,
        "fund": fund,
        "document": document,
        "page_start": page,
        "page_end": page,
        "section": section,
        "text": text,
    }


def test_prompt_augmenter_enforces_context_budget_and_attribution_metadata():
    retrieved = [
        evidence("1", "Cedar", "a" * 100, page=1),
        evidence("2", "Cedar", "b" * 100, page=2),
        evidence("3", "Cedar", "c" * 100, page=3),
    ]
    augmenter = PromptAugmenter(max_chunks=2, max_chars_per_chunk=30, max_total_evidence_chars=50)

    bundle = augmenter.build("What is the objective?", retrieved)

    assert list(bundle.evidence_map) == ["E1", "E2"]
    assert "Document: doc.pdf" in bundle.prompt
    assert "Page: 1" in bundle.prompt
    assert "[E3]" not in bundle.prompt
    # 30 characters from E1 + remaining 20 from E2.
    assert bundle.prompt.count("a") >= 30
    assert "b" * 20 in bundle.prompt
    assert "b" * 21 not in bundle.prompt


def test_answer_validation_rejects_hallucinated_or_insufficient_citations():
    evidence_map = {"E1": evidence("1", "Cedar", "support")}

    with pytest.raises(ValueError, match="not retrieved"):
        validate_answer(
            GeminiAnswer(status="supported", answer="x", evidence_ids=["E9"]),
            evidence_map,
        )

    with pytest.raises(ValueError, match="must cite at least one"):
        validate_answer(
            GeminiAnswer(status="supported", answer="x", evidence_ids=[]),
            evidence_map,
        )

    with pytest.raises(ValueError, match="at least two"):
        validate_answer(
            GeminiAnswer(status="conflicting", answer="x", evidence_ids=["E1"]),
            evidence_map,
        )


class StubRetriever:
    def __init__(self):
        self.registry = [
            {"canonical_name": "Cedar Fund", "aliases": ["Cedar Fund"]},
            {"canonical_name": "Oak Fund", "aliases": ["Oak Fund"]},
        ]

    def retrieve(self, query, funds=None, k=5, mode="auto"):
        fund = funds[0] if funds else None
        prefix = "c" if fund == "Cedar Fund" else "o"
        return [
            evidence(f"{prefix}1", fund, f"{fund} first evidence"),
            evidence(f"{prefix}2", fund, f"{fund} second evidence"),
        ][:k]


def test_multifund_generation_context_is_balanced_across_funds():
    assistant = GeminiFundAssistant(
        retriever=StubRetriever(),
        generator=object(),
        top_k=6,
        k_per_fund=2,
    )

    rows = assistant.retrieve("Compare Cedar Fund with Oak Fund")

    assert [row["chunk_id"] for row in rows] == ["c1", "o1", "c2", "o2"]


class StaticGenerator:
    def __init__(self, answer):
        self.answer_value = answer
        self.prompts = []

    def generate(self, prompt):
        self.prompts.append(prompt)
        raw = self.answer_value.model_dump_json()
        return self.answer_value, raw


class OneResultRetriever:
    registry = []

    def retrieve(self, query, funds=None, k=5, mode="auto"):
        return [evidence(
            "rights-1",
            None,
            "Fund units may be transferred freely.",
            document="11074.pdf",
            page=1,
            section="Прехвърляне на дялове",
        )]


def test_generation_maps_model_selected_evidence_back_to_real_source_metadata():
    model = StaticGenerator(GeminiAnswer(
        status="supported",
        answer="Yes. Fund units may be transferred freely.",
        evidence_ids=["E1"],
    ))
    assistant = GeminiFundAssistant(OneResultRetriever(), model, top_k=6)

    result = assistant.answer("Can I transfer my fund units to another person?", include_debug=True)

    assert result.status == "supported"
    assert result.answer.startswith("Yes")
    assert result.evidence_ids == ["E1"]
    assert result.sources == [{
        "evidence_id": "E1",
        "document": "11074.pdf",
        "page": 1,
        "section": "Прехвърляне на дялове",
        "fund": None,
        "chunk_id": "rights-1",
    }]
    assert "Fund units may be transferred freely" in model.prompts[0]


def test_empty_retrieval_returns_not_found_without_calling_generator():
    class EmptyRetriever:
        registry = []

        def retrieve(self, **kwargs):
            return []

    class MustNotRun:
        def generate(self, prompt):
            raise AssertionError("Generator should not be called without evidence")

    result = GeminiFundAssistant(EmptyRetriever(), MustNotRun()).answer("Unknown fact")
    assert result.status == "not_found"
    assert result.sources == []
