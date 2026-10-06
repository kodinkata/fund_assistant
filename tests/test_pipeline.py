from __future__ import annotations

import json
import os

import pytest

from fund_rag import retrieval
from fund_rag.chunking import chunk_documents
from fund_rag.fund_discovery import discover
from fund_rag.fund_segmentation import segment_documents
from fund_rag.generator import GeminiAnswer, GeminiFundAssistant, GeminiGenerator


class SimpleBM25:
    def __init__(self, tokenized_corpus):
        self.corpus = [set(tokens) for tokens in tokenized_corpus]

    def get_scores(self, query_tokens):
        query = set(query_tokens)
        return [float(len(query & document)) for document in self.corpus]


def lightweight_retriever(chunks, registry):
    """Exercise the real HybridRetriever ranking/routing code without model downloads."""
    obj = retrieval.HybridRetriever.__new__(retrieval.HybridRetriever)
    obj.chunks = chunks
    obj.registry = registry
    obj.chunk_languages = [
        retrieval.detect_language(retrieval.lexical_text(chunk))
        for chunk in chunks
    ]
    obj.lexical_texts = [retrieval.lexical_text(chunk) for chunk in chunks]
    obj.bm25 = SimpleBM25([
        retrieval.tokenize(text)
        for text in obj.lexical_texts
    ])
    return obj


class GroundedTestGenerator:
    """Deterministic provider double: generation still uses the real prompt/citation contract."""

    def generate(self, prompt):
        assert "Fund units may be transferred freely" in prompt
        answer = GeminiAnswer(
            status="supported",
            answer="Yes. Fund units may be transferred freely to another person, subject to applicable law.",
            evidence_ids=["E1"],
        )
        return answer, answer.model_dump_json()


def test_complete_rag_flow_question_to_retrieval_to_answer_with_sources(
    rights_document,
    single_fund_document,
):
    # 1. Fund discovery + document classification.
    documents = [rights_document, single_fund_document]
    registry, classifications, _ = discover(documents)

    # 2. Fund-aware segmentation.
    segments = segment_documents(documents, registry, classifications)

    # 3. Fund-safe chunking.
    chunks = chunk_documents(
        documents,
        segments,
        max_words=120,
        min_words=10,
        overlap_words=20,
    )

    # 4. Real HybridRetriever lexical routing/ranking logic.
    retriever = lightweight_retriever(chunks, registry)

    # 5. Real assistant orchestration + prompt augmentation + evidence validation.
    assistant = GeminiFundAssistant(
        retriever=retriever,
        generator=GroundedTestGenerator(),
        top_k=4,
        retrieval_mode="lexical",
    )

    result = assistant.answer("Can I transfer my fund units to another person?")

    # Assignment-required output contract.
    response = {
        "answer": result.answer,
        "funds": result.funds,
        "sources": [
            {
                "document": source["document"],
                "page": source["page"],
                "section": source["section"],
            }
            for source in result.sources
        ],
    }

    assert result.status == "supported"
    assert "transferred freely" in result.answer
    assert response["funds"] == []
    assert response["sources"] == [{
        "document": "investor_rights.pdf",
        "page": 1,
        "section": "Transfer of units",
    }]
    assert json.loads(json.dumps(response)) == response


@pytest.mark.live
@pytest.mark.skipif(
    os.environ.get("RUN_LIVE_RAG_TEST") != "1" or not os.environ.get("GEMINI_API_KEY"),
    reason="Set RUN_LIVE_RAG_TEST=1 and GEMINI_API_KEY to exercise the live Gemini provider.",
)
def test_live_gemini_generation_uses_supplied_evidence_only():
    """Optional network test for the actual Gemini adapter, excluded from normal CI."""
    generator = GeminiGenerator(
        api_key=os.environ["GEMINI_API_KEY"],
        model=os.environ.get("RAG_GENERATION_MODEL", "gemini-3.5-flash"),
    )
    prompt = """You are a grounded QA assistant. Use only the evidence.

Question: Can I transfer my fund units to another person?
Evidence:
[E1] Fund units may be transferred freely to another person, subject to applicable law.

Return a supported answer and cite E1.
"""

    answer, _ = generator.generate(prompt)

    assert answer.status == "supported"
    assert answer.evidence_ids == ["E1"]
    assert answer.answer.strip()
