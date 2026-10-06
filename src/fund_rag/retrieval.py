"""Multilingual retrieval over already-generated fund chunks.

Design:
- deterministic fund routing from the validated registry
- BGE-M3 dense retrieval for cross-language search
- language-aware BM25 for lexical precision
- auto mode: semantic for mixed-language pools, hybrid for monolingual pools
- optional diversified candidate generation + multilingual reranking

This module does not rerun ingestion, fund discovery, segmentation, or chunking.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path

CYRILLIC_RE = re.compile(r"[\u0400-\u04FF]")
LATIN_RE = re.compile(r"[A-Za-z]")
WORD_RE = re.compile(r"\w+(?:[&+.-]\w+)*", re.UNICODE)

DEFAULT_MODEL = "BAAI/bge-m3"
DEFAULT_RERANKER = "BAAI/bge-reranker-v2-m3"
INDEX_VERSION = 5


def clean(value: str) -> str:
    return " ".join((value or "").split())


def norm(value: str) -> str:
    return clean(unicodedata.normalize("NFKC", value or "")).casefold()


def tokenize(value: str) -> list[str]:
    return WORD_RE.findall(norm(value))


def detect_language(text: str) -> str:
    """Lightweight detector for this Bulgarian/English corpus."""
    cyrillic = len(CYRILLIC_RE.findall(text or ""))
    latin = len(LATIN_RE.findall(text or ""))
    if cyrillic > latin:
        return "bg"
    if latin > cyrillic:
        return "en"
    return "mixed"


def model_prefixes(model_name: str) -> tuple[str, str]:
    """E5 needs query:/passage: prefixes; BGE-M3 does not."""
    if "multilingual-e5" in model_name.casefold():
        return "query: ", "passage: "
    return "", ""


def semantic_text(chunk: dict) -> str:
    """Dense representation used for indexing and reranking."""
    section = clean(chunk.get("section", ""))
    text = clean(chunk.get("text", ""))
    return "\n".join(part for part in (section, text) if part)


def lexical_text(chunk: dict) -> str:
    """BM25 representation; fund names remain metadata, not repeated text."""
    return semantic_text(chunk)


def chunks_fingerprint(chunks: list[dict]) -> str:
    digest = hashlib.sha256()
    for chunk in chunks:
        for value in (
            chunk.get("chunk_id", ""),
            chunk.get("section", ""),
            chunk.get("text", ""),
        ):
            digest.update(str(value).encode("utf-8"))
            digest.update(b"\0")
    return digest.hexdigest()


def detect_query_funds(query: str, registry: list[dict]) -> list[str]:
    """Resolve known fund names/aliases; longest overlapping alias wins."""
    value = norm(query)
    matches = []

    for fund in registry:
        canonical = fund["canonical_name"]
        for alias in {canonical, *fund.get("aliases", [])}:
            alias_norm = norm(alias)
            if not alias_norm:
                continue

            start = value.find(alias_norm)
            while start != -1:
                matches.append({
                    "start": start,
                    "end": start + len(alias_norm),
                    "length": len(alias_norm),
                    "fund": canonical,
                })
                start = value.find(alias_norm, start + 1)

    selected = []
    for match in sorted(matches, key=lambda x: (-x["length"], x["start"])):
        overlaps = any(
            not (
                match["end"] <= chosen["start"]
                or match["start"] >= chosen["end"]
            )
            for chosen in selected
        )
        if not overlaps:
            selected.append(match)

    funds = []
    for match in sorted(selected, key=lambda x: x["start"]):
        if match["fund"] not in funds:
            funds.append(match["fund"])
    return funds


def strip_query_funds(
    query: str,
    registry: list[dict],
    funds: list[str],
) -> str:
    """Remove resolved fund names so ranking focuses on the information need."""
    wanted = set(funds)
    aliases = []

    for fund in registry:
        if fund["canonical_name"] in wanted:
            aliases.extend({
                fund["canonical_name"],
                *fund.get("aliases", []),
            })

    value = query
    for alias in sorted(
        {a for a in aliases if clean(a)},
        key=len,
        reverse=True,
    ):
        value = re.sub(re.escape(alias), " ", value, flags=re.IGNORECASE)

    value = re.sub(r"\s+", " ", value).strip(" \t\r\n,;:-")
    return value or query


def allowed_chunk_indices(
    chunks: list[dict],
    funds: list[str],
) -> set[int]:
    """Apply only validated chunk-level fund routing."""
    if not funds:
        return {
            i
            for i, chunk in enumerate(chunks)
            if chunk.get("fund") is None
        }

    wanted = set(funds)
    return {
        i
        for i, chunk in enumerate(chunks)
        if chunk.get("fund") in wanted
    }


def diversify_candidates(
    ranked: list[tuple[int, float]],
    chunk_languages: list[str],
    target_k: int = 50,
    global_k: int = 30,
    per_language_k: int = 10,
) -> list[tuple[int, float]]:
    """Union top global + top EN + top BG candidates, then fill to target_k."""
    if target_k <= 0:
        return []

    selected = []
    selected_ids = set()

    def add(item: tuple[int, float]) -> None:
        index, _ = item
        if index not in selected_ids and len(selected) < target_k:
            selected.append(item)
            selected_ids.add(index)

    for item in ranked[:global_k]:
        add(item)

    for language in ("en", "bg"):
        language_items = [
            item
            for item in ranked
            if chunk_languages[item[0]] == language
        ]
        for item in language_items[:per_language_k]:
            add(item)

    for item in ranked:
        if len(selected) >= target_k:
            break
        add(item)

    return selected


class MultilingualReranker:
    """Optional cross-encoder reranker, loaded lazily."""

    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER,
        device: str | None = None,
        batch_size: int = 4,
        max_length: int = 512,
    ):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def score(self, query: str, passages: list[str]) -> list[float]:
        scores = []
        torch = self.torch

        for start in range(0, len(passages), self.batch_size):
            batch = passages[start:start + self.batch_size]
            inputs = self.tokenizer(
                [query] * len(batch),
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            inputs = {
                key: value.to(self.device)
                for key, value in inputs.items()
            }
            with torch.no_grad():
                logits = self.model(**inputs, return_dict=True).logits.view(-1)
            scores.extend(logits.float().cpu().tolist())

        return scores


class HybridRetriever:
    def __init__(
        self,
        chunks: list[dict],
        registry: list[dict],
        index_dir: Path,
        model_name: str = DEFAULT_MODEL,
        rebuild: bool = False,
        batch_size: int = 4,
        device: str | None = None,
        reranker_model_name: str = DEFAULT_RERANKER,
        reranker_device: str | None = None,
        reranker_batch_size: int = 4,
    ):
        try:
            import faiss
            from rank_bm25 import BM25Okapi
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "Install sentence-transformers, faiss-cpu, rank-bm25, and transformers."
            ) from exc

        self.faiss = faiss
        self.chunks = chunks
        self.registry = registry
        self.index_dir = Path(index_dir)
        self.model_name = model_name
        self.batch_size = batch_size
        self.query_prefix, self.passage_prefix = model_prefixes(model_name)

        self.reranker_model_name = reranker_model_name
        self.reranker_device = reranker_device
        self.reranker_batch_size = reranker_batch_size
        self._reranker: MultilingualReranker | None = None

        self.index_dir.mkdir(parents=True, exist_ok=True)

        self.semantic_texts = [semantic_text(chunk) for chunk in chunks]
        self._chunk_index = {
            chunk.get("chunk_id"): i
            for i, chunk in enumerate(chunks)
            if chunk.get("chunk_id") is not None
        }
        self.lexical_texts = [lexical_text(chunk) for chunk in chunks]
        self.chunk_languages = [
            detect_language(text)
            for text in self.lexical_texts
        ]

        self.bm25 = BM25Okapi([
            tokenize(text)
            for text in self.lexical_texts
        ])

        print(f"[retrieval] Loading embedding model: {model_name}")
        
        self.model = (
            SentenceTransformer(model_name, device=device)
            if device
            else SentenceTransformer(model_name)
        )
        print("[retrieval] Embedding model loaded.")

        self.index = self._load_or_build_dense_index(rebuild)

    def _cache_metadata(self) -> dict:
        return {
            "index_version": INDEX_VERSION,
            "model_name": self.model_name,
            "passage_prefix": self.passage_prefix,
            "chunk_count": len(self.chunks),
            "chunk_fingerprint": chunks_fingerprint(self.chunks),
        }

    def _load_or_build_dense_index(self, rebuild: bool):
        index_path = self.index_dir / "semantic.faiss"
        metadata_path = self.index_dir / "semantic_index.json"
        expected = self._cache_metadata()
    
        print(f"[retrieval] rebuild={rebuild}")
        print(f"[retrieval] FAISS path: {index_path}")
        print(f"[retrieval] FAISS exists: {index_path.exists()}")
        print(f"[retrieval] metadata exists: {metadata_path.exists()}")
    
        if (
            not rebuild
            and index_path.exists()
            and metadata_path.exists()
        ):
            actual = json.loads(
                metadata_path.read_text(encoding="utf-8")
            )
    
            if actual == expected:
                print("[retrieval] Cache matches current chunks/model.")
                print("[retrieval] Loading existing FAISS index — NO re-embedding.")
    
                index = self.faiss.read_index(str(index_path))
    
                print(
                    f"[retrieval] Loaded existing FAISS index "
                    f"with {index.ntotal} vectors."
                )
    
                return index
    
            print("[retrieval] Saved index metadata DOES NOT match current state.")
            print("[retrieval] Differences:")
    
            for key in expected:
                if actual.get(key) != expected.get(key):
                    print(
                        f"  - {key}: "
                        f"saved={actual.get(key)!r} "
                        f"current={expected.get(key)!r}"
                    )
    
        elif rebuild:
            print("[retrieval] Explicit rebuild=True.")
    
        else:
            print("[retrieval] Existing FAISS cache is missing.")
    
        print(
            "[retrieval] REBUILDING embeddings + FAISS index. "
            "This is the expensive operation."
        )
    
        embeddings = self.model.encode(
            [
                self.passage_prefix + text
                for text in self.semantic_texts
            ],
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")
    
        index = self.faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
    
        self.faiss.write_index(index, str(index_path))
    
        metadata_path.write_text(
            json.dumps(
                expected,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    
        print(
            f"[retrieval] New FAISS index saved with "
            f"{index.ntotal} vectors."
        )
    
        return index
        
    def _resolve_query(
        self,
        query: str,
        funds: list[str] | None,
    ) -> tuple[list[str], set[int], str, str]:
        query_funds = (
            list(funds)
            if funds is not None
            else detect_query_funds(query, self.registry)
        )
        allowed = allowed_chunk_indices(self.chunks, query_funds)
        search_query = (
            strip_query_funds(query, self.registry, query_funds)
            if query_funds
            else query
        )
        return (
            query_funds,
            allowed,
            search_query,
            detect_language(search_query),
        )

    def _semantic_rank(
        self,
        query: str,
        allowed: set[int],
        pool_size: int,
    ) -> list[tuple[int, float]]:
        vector = self.model.encode(
            [self.query_prefix + query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

        scores, indices = self.index.search(
            vector,
            len(self.chunks),
        )

        ranked = []
        for index, score in zip(indices[0], scores[0]):
            index = int(index)
            if index >= 0 and index in allowed:
                ranked.append((index, float(score)))
                if len(ranked) >= pool_size:
                    break
        return ranked

    def _lexical_rank(
        self,
        query: str,
        allowed: set[int],
        pool_size: int,
        query_language: str,
    ) -> list[tuple[int, float]]:
        lexical_allowed = (
            allowed
            if query_language == "mixed"
            else {
                i
                for i in allowed
                if self.chunk_languages[i] in {query_language, "mixed"}
            }
        )

        scores = self.bm25.get_scores(tokenize(query))
        return sorted(
            (
                (i, float(scores[i]))
                for i in lexical_allowed
                if scores[i] > 0
            ),
            key=lambda pair: pair[1],
            reverse=True,
        )[:pool_size]

    @staticmethod
    def _rrf(
        semantic: list[tuple[int, float]],
        lexical: list[tuple[int, float]],
        semantic_weight: float,
        lexical_weight: float,
        rrf_k: int,
    ):
        fused = {}
        semantic_ranks = {}
        lexical_ranks = {}

        for rank, (index, _) in enumerate(semantic, start=1):
            semantic_ranks[index] = rank
            fused[index] = (
                fused.get(index, 0.0)
                + semantic_weight / (rrf_k + rank)
            )

        for rank, (index, _) in enumerate(lexical, start=1):
            lexical_ranks[index] = rank
            fused[index] = (
                fused.get(index, 0.0)
                + lexical_weight / (rrf_k + rank)
            )

        return fused, semantic_ranks, lexical_ranks

    def _candidate_languages(self, allowed: set[int]) -> set[str]:
        return {
            self.chunk_languages[i]
            for i in allowed
            if self.chunk_languages[i] in {"en", "bg"}
        }

    def _effective_mode(
        self,
        requested_mode: str,
        allowed: set[int],
    ) -> str:
        if requested_mode != "auto":
            return requested_mode

        # BM25 is useful within one language, but cannot bridge EN <-> BG.
        # For mixed-language pools, BGE-M3 semantic retrieval is the
        # evaluated multilingual path.
        return (
            "semantic"
            if len(self._candidate_languages(allowed)) > 1
            else "hybrid"
        )

    def _get_reranker(self) -> MultilingualReranker:
        if self._reranker is None:
            self._reranker = MultilingualReranker(
                model_name=self.reranker_model_name,
                device=self.reranker_device,
                batch_size=self.reranker_batch_size,
            )
        return self._reranker

    def candidate_set(
        self,
        query: str,
        funds: list[str] | None = None,
        candidate_k: int = 50,
        global_k: int = 30,
        per_language_k: int = 10,
    ) -> list[dict]:
        """High-recall BGE candidates for optional reranking/evaluation."""
        (
            query_funds,
            allowed,
            search_query,
            query_language,
        ) = self._resolve_query(query, funds)

        if not allowed:
            return []

        dense = self._semantic_rank(
            search_query,
            allowed,
            pool_size=len(allowed),
        )
        dense_ranks = {
            index: rank
            for rank, (index, _) in enumerate(dense, start=1)
        }

        candidates = diversify_candidates(
            dense,
            self.chunk_languages,
            target_k=min(candidate_k, len(dense)),
            global_k=global_k,
            per_language_k=per_language_k,
        )

        rows = []
        for candidate_rank, (index, score) in enumerate(
            candidates,
            start=1,
        ):
            item = dict(self.chunks[index])
            item.update({
                "candidate_rank": candidate_rank,
                "candidate_dense_rank": dense_ranks[index],
                "candidate_language": self.chunk_languages[index],
                "query_language": query_language,
                "query_funds": query_funds,
                "search_query": search_query,
                "semantic_score": score,
            })
            rows.append(item)
        return rows

    def retrieve(
        self,
        query: str,
        k: int = 5,
        mode: str = "auto",
        funds: list[str] | None = None,
        pool_size: int = 100,
        semantic_weight: float = 3.0,
        lexical_weight: float = 1.0,
        rrf_k: int = 20,
        candidate_k: int = 50,
        global_k: int = 30,
        per_language_k: int = 10,
    ) -> list[dict]:
        if mode not in {
            "auto",
            "semantic",
            "lexical",
            "hybrid",
            "reranked",
        }:
            raise ValueError(
                "mode must be auto, semantic, lexical, hybrid, or reranked"
            )

        (
            query_funds,
            allowed,
            search_query,
            query_language,
        ) = self._resolve_query(query, funds)

        if not allowed:
            return []

        effective_mode = self._effective_mode(mode, allowed)

        semantic = []
        lexical = []
        semantic_ranks = {}
        lexical_ranks = {}
        fused = {}
        dense_ranks = {}
        reranker_scores = {}

        if effective_mode == "reranked":
            candidates = self.candidate_set(
                query,
                funds=query_funds,
                candidate_k=candidate_k,
                global_k=global_k,
                per_language_k=per_language_k,
            )
            candidate_pairs = []
            for item in candidates:
                index = self._chunk_index.get(item.get("chunk_id"))
                if index is not None:
                    candidate_pairs.append((index, item))

            scores = self._get_reranker().score(
                search_query,
                [
                    self.semantic_texts[index]
                    for index, _ in candidate_pairs
                ],
            )

            reranked = sorted(
                [
                    (index, float(score))
                    for (index, _), score
                    in zip(candidate_pairs, scores)
                ],
                key=lambda pair: pair[1],
                reverse=True,
            )
            ordered = [index for index, _ in reranked]
            reranker_scores = dict(reranked)
            dense_ranks = {
                index: item["candidate_dense_rank"]
                for index, item in candidate_pairs
            }
            semantic = [
                (index, item["semantic_score"])
                for index, item in candidate_pairs
            ]

        else:
            if effective_mode in {"semantic", "hybrid"}:
                semantic = self._semantic_rank(
                    search_query,
                    allowed,
                    pool_size,
                )
                semantic_ranks = {
                    index: rank
                    for rank, (index, _) in enumerate(
                        semantic,
                        start=1,
                    )
                }

            if effective_mode in {"lexical", "hybrid"}:
                lexical = self._lexical_rank(
                    search_query,
                    allowed,
                    pool_size,
                    query_language,
                )
                lexical_ranks = {
                    index: rank
                    for rank, (index, _) in enumerate(
                        lexical,
                        start=1,
                    )
                }

            if effective_mode == "semantic":
                ordered = [index for index, _ in semantic]
            elif effective_mode == "lexical":
                ordered = [index for index, _ in lexical]
            else:
                (
                    fused,
                    semantic_ranks,
                    lexical_ranks,
                ) = self._rrf(
                    semantic,
                    lexical,
                    semantic_weight,
                    lexical_weight,
                    rrf_k,
                )
                ordered = sorted(
                    fused,
                    key=fused.get,
                    reverse=True,
                )

        semantic_scores = dict(semantic)
        lexical_scores = dict(lexical)
        results = []

        for rank, index in enumerate(ordered[:k], start=1):
            item = dict(self.chunks[index])
            item.update({
                "rank": rank,
                "query_funds": query_funds,
                "query_language": query_language,
                "candidate_language": self.chunk_languages[index],
                "route": "no_fund" if not query_funds else "fund",
                "search_query": search_query,
                "requested_mode": mode,
                "effective_mode": effective_mode,
                "semantic_score": semantic_scores.get(index),
                "bm25_score": lexical_scores.get(index),
            })

            if effective_mode == "semantic":
                item["semantic_rank"] = semantic_ranks.get(index)

            elif effective_mode == "hybrid":
                item.update({
                    "hybrid_score": fused[index],
                    "semantic_rank": semantic_ranks.get(index),
                    "lexical_rank": lexical_ranks.get(index),
                })

            elif effective_mode == "reranked":
                item.update({
                    "candidate_dense_rank": dense_ranks.get(index),
                    "reranker_score": reranker_scores.get(index),
                })

            results.append(item)

        return results

    def retrieve_per_fund(
        self,
        query: str,
        funds: list[str] | None = None,
        k_per_fund: int = 3,
        mode: str = "auto",
        **kwargs,
    ) -> list[dict]:
        """Retrieve independently per fund so comparisons stay balanced."""
        funds = (
            funds
            if funds is not None
            else detect_query_funds(query, self.registry)
        )

        results = []
        for fund in funds:
            results.extend(
                self.retrieve(
                    query=query,
                    k=k_per_fund,
                    mode=mode,
                    funds=[fund],
                    **kwargs,
                )
            )
        return results
