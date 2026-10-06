"""Gemini generator + grounded prompt augmentation for Fund RAG.

This module does not change ingestion, chunking, fund routing, embeddings,
FAISS, BM25, or retrieval evaluation.

Runtime:
    query
      -> existing HybridRetriever
      -> prompt augmentation
      -> Gemini API
      -> structured grounded answer
      -> deterministic citation validation

Default generator:
    gemini-3.5-flash

Requires:
    pip install google-genai pydantic
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from fund_rag import retrieval


DEFAULT_GEMINI_MODEL = "gemini-3.5-flash"


class GeminiAnswer(BaseModel):
    status: Literal[
        "supported",
        "not_found",
        "ambiguous",
        "conflicting",
    ] = Field(
        description=(
            "supported when evidence answers the question; not_found when it "
            "does not; ambiguous when the question/evidence cannot uniquely "
            "resolve the request; conflicting when supplied sources disagree."
        )
    )
    answer: str = Field(
        description="Concise natural-language answer in the user's language."
    )
    evidence_ids: list[str] = Field(
        description=(
            "Only evidence IDs such as E1 or E2 that directly support the answer. "
            "Use an empty list for not_found when nothing supports an answer."
        )
    )


SYSTEM_INSTRUCTIONS = """You are a fund-documentation question answering assistant.

Use ONLY the supplied evidence. Do not use outside knowledge.

Requirements:
1. Keep facts for different funds separate.
2. Answer in the same language as the user's question unless explicitly asked otherwise.
3. If the evidence does not contain the requested information, use status="not_found".
4. If the question is genuinely ambiguous and the evidence cannot resolve it, use status="ambiguous".
5. If supplied evidence materially disagrees, use status="conflicting" and explain the disagreement.
6. Never invent a fund, fact, number, document, page, section, or citation.
7. Cite only evidence IDs that directly support the answer.
8. A supported answer must cite at least one evidence ID.
9. Be concise but include all information necessary to answer the question.
"""


@dataclass
class PromptBundle:
    query: str
    prompt: str
    evidence_map: dict[str, dict]


@dataclass
class AssistantResult:
    query: str
    status: str
    answer: str
    funds: list[str]
    sources: list[dict]
    evidence_ids: list[str]
    retrieved: list[dict]
    prompt: str | None = None
    raw_model_output: str | None = None

    def to_dict(
        self,
        include_retrieved: bool = False,
        include_debug: bool = False,
    ) -> dict:
        payload = {
            "query": self.query,
            "status": self.status,
            "answer": self.answer,
            "funds": self.funds,
            "sources": self.sources,
            "evidence_ids": self.evidence_ids,
        }
        if include_retrieved:
            payload["retrieved"] = self.retrieved
        if include_debug:
            payload["prompt"] = self.prompt
            payload["raw_model_output"] = self.raw_model_output
        return payload


def page_label(item: dict) -> str | int | None:
    start = item.get("page_start")
    end = item.get("page_end")

    if start is None and end is None:
        return item.get("page")
    if start is None:
        return end
    if end is None or start == end:
        return start
    return f"{start}-{end}"


def dedupe_results(results: list[dict]) -> list[dict]:
    seen = set()
    output = []

    for item in results:
        key = item.get("chunk_id") or (
            item.get("document"),
            item.get("page_start"),
            item.get("page_end"),
            item.get("section"),
            item.get("text"),
        )
        if key not in seen:
            seen.add(key)
            output.append(item)

    return output


class PromptAugmenter:
    """Convert retrieved chunks into attributable evidence for generation."""

    def __init__(
        self,
        max_chunks: int = 6,
        max_chars_per_chunk: int = 4500,
        max_total_evidence_chars: int = 22000,
    ):
        self.max_chunks = max_chunks
        self.max_chars_per_chunk = max_chars_per_chunk
        self.max_total_evidence_chars = max_total_evidence_chars

    def build(
        self,
        query: str,
        retrieved: list[dict],
    ) -> PromptBundle:
        evidence_map: dict[str, dict] = {}
        blocks = []
        used_chars = 0

        for item in dedupe_results(retrieved)[: self.max_chunks]:
            if used_chars >= self.max_total_evidence_chars:
                break

            text = (item.get("text") or "").strip()
            remaining = self.max_total_evidence_chars - used_chars
            text = text[: min(self.max_chars_per_chunk, remaining)].rstrip()

            if not text:
                continue

            evidence_id = f"E{len(evidence_map) + 1}"
            evidence_map[evidence_id] = item

            blocks.append(
                f"[{evidence_id}]\n"
                f"Fund: {item.get('fund')}\n"
                f"Document: {item.get('document')}\n"
                f"Page: {page_label(item)}\n"
                f"Section: {item.get('section')}\n"
                f"Text:\n{text}"
            )
            used_chars += len(text)

        evidence_text = "\n\n".join(blocks) or "[NO EVIDENCE RETRIEVED]"

        prompt = (
            f"{SYSTEM_INSTRUCTIONS}\n\n"
            f"QUESTION:\n{query}\n\n"
            f"EVIDENCE:\n{evidence_text}"
        )

        return PromptBundle(
            query=query,
            prompt=prompt,
            evidence_map=evidence_map,
        )


class GeminiGenerator:
    """Small provider adapter around the Google Gen AI SDK."""

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_GEMINI_MODEL,
    ):
        if not api_key:
            raise ValueError("A Gemini API key is required.")

        from google import genai

        self.client = genai.Client(api_key=api_key)
        self.model = model

    def generate(self, prompt: str) -> tuple[GeminiAnswer, str]:
        from google.genai import types

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=GeminiAnswer,
            ),
        )

        raw_text = response.text or ""
        parsed = GeminiAnswer.model_validate_json(raw_text)
        return parsed, raw_text


def validate_answer(
    answer: GeminiAnswer,
    evidence_map: dict[str, dict],
) -> GeminiAnswer:
    """Reject citations that were not actually supplied to Gemini."""
    unique_ids = list(dict.fromkeys(answer.evidence_ids))
    unknown = [
        evidence_id
        for evidence_id in unique_ids
        if evidence_id not in evidence_map
    ]

    if unknown:
        raise ValueError(
            "Gemini cited evidence that was not retrieved: "
            + ", ".join(unknown)
        )

    if answer.status == "supported" and not unique_ids:
        raise ValueError(
            "A supported answer must cite at least one retrieved evidence item."
        )

    if answer.status == "conflicting" and len(unique_ids) < 2:
        raise ValueError(
            "A conflicting answer must cite at least two competing evidence items."
        )

    return answer.model_copy(update={"evidence_ids": unique_ids})


class GeminiFundAssistant:
    """Complete runtime path: retrieval -> augmentation -> Gemini -> citations."""

    def __init__(
        self,
        retriever: Any,
        generator: GeminiGenerator,
        augmenter: PromptAugmenter | None = None,
        top_k: int = 6,
        retrieval_mode: str = "auto",
        k_per_fund: int = 4,
    ):
        self.retriever = retriever
        self.generator = generator
        self.augmenter = augmenter or PromptAugmenter()
        self.top_k = top_k
        self.retrieval_mode = retrieval_mode
        self.k_per_fund = k_per_fund

    def retrieve(self, query: str) -> list[dict]:
        funds = retrieval.detect_query_funds(
            query,
            self.retriever.registry,
        )

        if len(funds) > 1:
            search_query = retrieval.strip_query_funds(
                query,
                self.retriever.registry,
                funds,
            )

            # Retrieve independently so one fund cannot dominate a comparison.
            # Interleave by per-fund rank before the prompt context budget is applied.
            per_fund = [
                self.retriever.retrieve(
                    query=search_query,
                    funds=[fund],
                    k=self.k_per_fund,
                    mode=self.retrieval_mode,
                )
                for fund in funds
            ]
            balanced = [
                items[rank]
                for rank in range(self.k_per_fund)
                for items in per_fund
                if rank < len(items)
            ]
            return dedupe_results(balanced)

        return dedupe_results(
            self.retriever.retrieve(
                query=query,
                funds=funds or None,
                k=self.top_k,
                mode=self.retrieval_mode,
            )
        )

    def answer(
        self,
        query: str,
        include_debug: bool = False,
    ) -> AssistantResult:
        retrieved = self.retrieve(query)
        bundle = self.augmenter.build(query, retrieved)

        if not bundle.evidence_map:
            return AssistantResult(
                query=query,
                status="not_found",
                answer=(
                    "The requested information was not found in the provided documents."
                ),
                funds=[],
                sources=[],
                evidence_ids=[],
                retrieved=[],
                prompt=bundle.prompt if include_debug else None,
                raw_model_output=None,
            )

        model_answer, raw_text = self.generator.generate(bundle.prompt)
        model_answer = validate_answer(
            model_answer,
            bundle.evidence_map,
        )

        sources = []
        for evidence_id in model_answer.evidence_ids:
            item = bundle.evidence_map[evidence_id]
            sources.append({
                "evidence_id": evidence_id,
                "document": item.get("document"),
                "page": page_label(item),
                "section": item.get("section"),
                "fund": item.get("fund"),
                "chunk_id": item.get("chunk_id"),
            })

        funds = list(dict.fromkeys(
            source["fund"]
            for source in sources
            if source.get("fund")
        ))

        return AssistantResult(
            query=query,
            status=model_answer.status,
            answer=model_answer.answer,
            funds=funds,
            sources=sources,
            evidence_ids=model_answer.evidence_ids,
            retrieved=retrieved,
            prompt=bundle.prompt if include_debug else None,
            raw_model_output=raw_text if include_debug else None,
        )


def load_retriever(
    root: str | Path,
    device: str | None = None,
):
    """Load the already-built retrieval artifacts.

    semantic_index.json is used to recover the exact embedding model used
    when the FAISS index was built.
    """
    root = Path(root)

    chunk_path = root / "artifacts" / "chunks" / "chunks.json"
    registry_path = (
        root
        / "artifacts"
        / "fund_discovery"
        / "registry.json"
    )
    index_dir = root / "artifacts" / "retrieval_bge_m3"
    faiss_path = index_dir / "semantic.faiss"
    metadata_path = index_dir / "semantic_index.json"

    required = [
        chunk_path,
        registry_path,
        faiss_path,
        metadata_path,
    ]

    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Run retrieval/indexing first. Missing:\n- "
            + "\n- ".join(missing)
        )

    chunks = json.loads(chunk_path.read_text(encoding="utf-8"))
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    metadata = json.loads(
        metadata_path.read_text(encoding="utf-8")
    )

    return retrieval.HybridRetriever(
        chunks=chunks,
        registry=registry,
        index_dir=index_dir,
        model_name=metadata["model_name"],
        rebuild=False,
        batch_size=4,
        device=device,
    )


def build_assistant(
    root: str | Path,
    api_key: str,
    model: str = DEFAULT_GEMINI_MODEL,
    retrieval_device: str | None = None,
    top_k: int = 6,
) -> GeminiFundAssistant:
    retriever = load_retriever(
        root=root,
        device=retrieval_device,
    )
    generator = GeminiGenerator(
        api_key=api_key,
        model=model,
    )
    return GeminiFundAssistant(
        retriever=retriever,
        generator=generator,
        top_k=top_k,
        retrieval_mode="auto",
    )
