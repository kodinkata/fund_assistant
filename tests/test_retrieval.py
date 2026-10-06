from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from fund_rag import retrieval


def test_query_fund_resolution_prefers_longest_overlapping_name():
    registry = [
        {"canonical_name": "K&H egészség", "aliases": ["K&H egészség"]},
        {"canonical_name": "K&H egészség 2", "aliases": ["K&H egészség 2"]},
    ]

    funds = retrieval.detect_query_funds(
        "What is the objective of K&H egészség 2?",
        registry,
    )

    assert funds == ["K&H egészség 2"]


def test_chunk_routing_keeps_general_and_named_fund_content_separate():
    chunks = [
        {"fund": None, "text": "Investor rights"},
        {"fund": "Cedar", "text": "Cedar objective"},
        {"fund": "Oak", "text": "Oak objective"},
    ]

    assert retrieval.allowed_chunk_indices(chunks, []) == {0}
    assert retrieval.allowed_chunk_indices(chunks, ["Cedar"]) == {1}
    assert retrieval.allowed_chunk_indices(chunks, ["Cedar", "Oak"]) == {1, 2}


def test_auto_mode_uses_semantic_for_mixed_language_pool_and_hybrid_for_single_language():
    obj = retrieval.HybridRetriever.__new__(retrieval.HybridRetriever)
    obj.chunk_languages = ["en", "bg", "en"]

    assert obj._effective_mode("auto", {0, 1}) == "semantic"
    assert obj._effective_mode("auto", {0, 2}) == "hybrid"
    assert obj._effective_mode("lexical", {0, 1}) == "lexical"


class CountingModel:
    def __init__(self):
        self.calls = 0

    def encode(self, texts, **kwargs):
        self.calls += 1
        rows = []
        for i, _ in enumerate(texts):
            rows.append([1.0, float(i + 1), 0.5])
        return np.asarray(rows, dtype="float32")


class FakeIndex:
    def __init__(self, dim):
        self.dim = dim
        self.ntotal = 0

    def add(self, embeddings):
        self.ntotal += len(embeddings)


class FakeFaiss:
    IndexFlatIP = FakeIndex

    @staticmethod
    def write_index(index, path):
        Path(path).write_text(json.dumps({"dim": index.dim, "ntotal": index.ntotal}))

    @staticmethod
    def read_index(path):
        data = json.loads(Path(path).read_text())
        index = FakeIndex(data["dim"])
        index.ntotal = data["ntotal"]
        return index


def test_dense_index_cache_reuses_matching_embeddings(tmp_path):
    chunks = [
        {"chunk_id": "1", "section": "A", "text": "alpha"},
        {"chunk_id": "2", "section": "B", "text": "beta"},
    ]
    obj = retrieval.HybridRetriever.__new__(retrieval.HybridRetriever)
    obj.faiss = FakeFaiss()
    obj.chunks = chunks
    obj.index_dir = tmp_path
    obj.model_name = "BAAI/bge-m3"
    obj.passage_prefix = ""
    obj.semantic_texts = [retrieval.semantic_text(c) for c in chunks]
    obj.batch_size = 4
    obj.model = CountingModel()

    first = obj._load_or_build_dense_index(rebuild=False)
    second = obj._load_or_build_dense_index(rebuild=False)

    assert first.ntotal == 2
    assert second.ntotal == 2
    assert obj.model.calls == 1, "Matching cache metadata must prevent re-embedding."
    metadata = json.loads((tmp_path / "semantic_index.json").read_text(encoding="utf-8"))
    assert metadata["model_name"] == "BAAI/bge-m3"
    assert metadata["chunk_count"] == 2


def test_real_retrieve_logic_applies_fund_filter_and_lexical_ranking(simple_bm25_cls):
    chunks = [
        {"chunk_id": "g", "fund": None, "section": "Rights", "text": "Fund units may be transferred freely."},
        {"chunk_id": "c", "fund": "Cedar Global Fund", "section": "Objective", "text": "Cedar seeks growth."},
        {"chunk_id": "o", "fund": "Oak Income Fund", "section": "Objective", "text": "Oak seeks income."},
    ]
    registry = [
        {"canonical_name": "Cedar Global Fund", "aliases": ["Cedar Global Fund"]},
        {"canonical_name": "Oak Income Fund", "aliases": ["Oak Income Fund"]},
    ]

    obj = retrieval.HybridRetriever.__new__(retrieval.HybridRetriever)
    obj.chunks = chunks
    obj.registry = registry
    obj.chunk_languages = [retrieval.detect_language(retrieval.lexical_text(c)) for c in chunks]
    obj.lexical_texts = [retrieval.lexical_text(c) for c in chunks]
    obj.bm25 = simple_bm25_cls([retrieval.tokenize(text) for text in obj.lexical_texts])

    general = obj.retrieve("Can I transfer my fund units?", k=3, mode="lexical")
    cedar = obj.retrieve("What is the objective of Cedar Global Fund?", k=3, mode="lexical")

    assert [row["chunk_id"] for row in general] == ["g"]
    assert [row["chunk_id"] for row in cedar] == ["c"]
    assert cedar[0]["query_funds"] == ["Cedar Global Fund"]
    assert "Cedar Global Fund" not in cedar[0]["search_query"]
