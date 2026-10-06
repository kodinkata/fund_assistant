# Fund Documentation Assistant

A multilingual Retrieval-Augmented Generation (RAG) system for answering questions from the supplied investment-fund documentation while preserving fund boundaries, source provenance, and grounded citations.

The implementation is intentionally small and inspectable. Each pipeline phase is implemented as a plain Python module and evaluated in a corresponding notebook. Generated artifacts are persisted between phases so the expensive work is not repeated at query time.

## What the system does

Given a question such as:

> Can I transfer my fund units to another person?

or:

> Какъв е минималният процент устойчиви инвестиции за ОББ ЕкспертИйз Дефанзивен Консервативен Отговорно Инвестиращ?

The runtime path is:

```text
User question
    │
    ▼
Fund-name routing
    │
    ▼
BGE-M3 dense retrieval + language-aware BM25 policy
    │
    ▼
Top evidence chunks with document/fund/page/section metadata
    │
    ▼
Bounded prompt augmentation (E1, E2, ...)
    │
    ▼
Gemini structured generation
    │
    ▼
Python validates evidence IDs and maps them back to real metadata
    │
    ▼
Answer + funds + document/page/section sources
```

Example output shape:

```json
{
  "answer": "...",
  "funds": [
    "Fund name"
  ],
  "sources": [
    {
      "document": "document_name.pdf",
      "page": 12,
      "section": "Investment strategy"
    }
  ]
}
```

This matches the assignment requirement while keeping citation metadata outside the model's control.

---

## Design principles

The implementation follows four rules throughout the pipeline:

1. **Fund ownership is resolved before retrieval.** Chunks are created inside validated fund segments, so one chunk cannot silently mix two sub-funds.
2. **The generator does not search the corpus.** Retrieval selects candidate evidence; Gemini only reasons over the bounded evidence supplied to it.
3. **The model does not invent citation locations.** Gemini cites evidence IDs such as `E2`; Python maps `E2` back to the chunk's stored document/page/section metadata.
4. **Each phase is evaluated independently.** Parsing quality, fund discovery, segmentation, chunking, retrieval, and generation can therefore fail independently and be diagnosed independently.

The notebooks are development/evaluation notebooks; the `.py` modules contain the reusable pipeline logic.

---

# Architecture

## Offline indexing path

```text
data/documents/*.pdf
        │
        ▼
1. ingestion.py
   PyMuPDF text/layout/tables/images
   + vector-chart extraction / DePlot
        │
        ▼
artifacts/ingestion/*.json
        │
        ▼
2. fund_discovery.py
   full-corpus high-precision fund registry
        │
        ├── registry.json
        ├── classifications.json
        └── candidates.json
        │
        ▼
3. fund_segmentation.py
   structural ownership ranges
        │
        ▼
artifacts/fund_segmentation/segments.json
        │
        ▼
4. chunking.py
   heading-aware chunks inside segment boundaries
        │
        ▼
artifacts/chunks/chunks.json
        │
        ▼
5. retrieval.py
   BGE-M3 embeddings + FAISS
        │
        ├── semantic.faiss
        └── semantic_index.json
```

## Online question-answering path

```text
question
  │
  ├── detect known fund names from registry
  │
  ├── filter eligible chunks by fund ownership
  │
  ├── semantic / hybrid retrieval (mode="auto")
  │
  ├── bounded prompt construction
  │
  ├── Gemini structured answer
  │
  └── deterministic citation validation
```

The expensive document embeddings are therefore **not recomputed for every question**. FAISS is loaded from disk when its saved metadata matches the current chunks/model fingerprint. BM25 is inexpensive and is rebuilt in memory from the saved chunks at startup.

---

# Repository layout

```text
fund-rag/
├── data/
│   └── documents/                  # six supplied source PDFs
├── notebooks/
│   ├── ingestion_eval.ipynb
│   ├── fund_discovery_eval.ipynb
│   ├── fund_segmentation_eval.ipynb
│   ├── chunking_eval.ipynb
│   ├── retrieval_eval.ipynb
│   ├── generator_eval.ipynb
│   └── assistant.ipynb
├── src/fund_rag/
│   ├── ingestion.py
│   ├── fund_discovery.py
│   ├── fund_segmentation.py
│   ├── chunking.py
│   ├── retrieval.py
│   └── generator.py
├── tests/                           # automated tests; final refresh follows phase review
├── pyproject.toml
└── README.md
```

`artifacts/` and `.cache/` are generated locally and should not be committed.

---

# Setup

Python 3.11+ is required.

## Windows PowerShell

From the repository root:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -U pip
.\.venv\Scripts\python.exe -m pip install -e ".[notebooks,test]"
.\.venv\Scripts\python.exe -m jupyter lab
```

## macOS / Linux

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -U pip
.venv/bin/python -m pip install -e ".[notebooks,test]"
.venv/bin/python -m jupyter lab
```

The first ingestion/retrieval runs may download model weights for DePlot and BGE-M3. Those models are cached under the project Hugging Face cache configured by the notebooks.

## Gemini API key

Do not place an API key in a notebook or commit it to Git. The assistant first checks `GEMINI_API_KEY` and otherwise asks for it through `getpass`, which does not echo the value.

PowerShell example:

```powershell
$env:GEMINI_API_KEY="your-key"
```

The current generator uses `gemini-3.5-flash` through the Google Gen AI SDK. The model can be changed without touching the RAG logic by setting `RAG_GENERATION_MODEL`.

---

# Recommended execution order

Run the notebooks in this order:

1. `ingestion_eval.ipynb`
2. `fund_discovery_eval.ipynb`
3. `fund_segmentation_eval.ipynb`
4. `chunking_eval.ipynb`
5. `retrieval_eval.ipynb`
6. `generator_eval.ipynb`
7. `assistant.ipynb`

This order mirrors the artifact dependencies. A downstream notebook loads artifacts from the previous phase instead of silently rerunning upstream logic.

---

# Phase 1 — ingestion and parsing

**Implementation:** `src/fund_rag/ingestion.py`  
**Evaluation:** `notebooks/ingestion_eval.ipynb`

## Approach

The parser uses **PyMuPDF** as the primary document engine. For each page it preserves:

- native text;
- ordered text blocks;
- block bounding boxes;
- font-size and boldness signals used later for headings;
- detected tables and reconstructed rows;
- embedded images;
- vector-chart regions;
- a normalized `text_for_rag` representation.

Text blocks remain structured rather than being flattened immediately because fund segmentation and heading-aware chunking need layout information later.

### Tables

PyMuPDF's table detection is used to preserve row structure. Tables are not converted to screenshots and re-read by an LLM. This keeps table content deterministic and cheap where native PDF structure is available.

### Charts

Vector drawings are clustered and rendered as chart crops. **DePlot** converts a detected chart into underlying tabular rows. Those rows become text that can later be retrieved.

This is intentionally more targeted than running a general multimodal model over every page. It is sufficient for the supplied performance chart while keeping the pipeline understandable.

### Images

Embedded images are extracted and preserved. The current pipeline does **not** attempt general semantic image understanding. That is a deliberate boundary: preserving an image reliably and interpreting every possible image are different problems.

## Evaluation result

The last validated six-document run reported:

| Ingestion metric | Result |
|---|---:|
| Document coverage | 100% |
| Page coverage | 100% |
| Gold text phrase recall | 100% |
| Gold table-page detection recall | 100% |
| Gold table-row exact recall | 100% |
| Gold chart detection recall | 100% |
| Chart numeric-point accuracy | 100% |
| Precision on known-negative chart pages | 100% |
| Embedded-image extraction recall | 100% |
| Image decode success | 100% |
| Retrieval-modality inclusion | 100% |

The benchmark includes manually checked phrases, table rows, image pages, negative chart pages, and the known annual-performance chart. It is more meaningful than merely asserting that each PDF produced some text.

## Limitations and possible improvements

- **No OCR fallback.** Image-only/scanned PDFs would need OCR before they could enter the current retrieval path.
- **Chart interpretation is specialized.** DePlot is useful for plot-to-table extraction but is not a universal visual-document understanding system.
- **Heavy first run.** DePlot requires model weights and is only loaded when a chart is detected, but chart pages still make ingestion heavier than plain PyMuPDF parsing.
- **Artifact invalidation is simple.** A production ingestion service should fingerprint source bytes and parser/model versions and only reprocess changed documents.
- **Images are preserved, not understood.** If important fund facts move into infographics, a multimodal extraction phase would be needed.

---

# Phase 2 — fund discovery and document classification

**Implementation:** `src/fund_rag/fund_discovery.py`  
**Evaluation:** `notebooks/fund_discovery_eval.ipynb`

## Why this phase exists

The corpus contains both standalone fund documents and a large umbrella prospectus containing many sub-funds. Searching the entire prospectus without resolving ownership can return a perfectly relevant sentence for the **wrong fund**.

Fund discovery therefore happens before chunking.

## Discovery strategy

The production code scans the **full parsed corpus** and accepts fund-name candidates only from high-precision structural evidence:

- explicit linguistic relations such as “X is a sub-fund”;
- explicit `Product Name`, `Fund Name`, or `Sub-fund Name` labels;
- fund-list tables;
- strongly supported first-page title patterns.

Canonicalization is deliberately conservative. Similar names such as `K&H egészség` and `K&H egészség 2` must remain separate entities.

## Evaluation result

The last validated full-corpus run produced:

- **41 expected funds discovered**;
- **0 missing funds**;
- **0 extra funds**;
- **100% precision**;
- **100% recall**;
- **100% F1**.

It also classified the documents as expected:

- investor-rights document → `no_fund`;
- standalone disclosures/KID → `single_fund`;
- umbrella prospectus → `multi_fund`.

## Limitations and possible improvements

- The strategy intentionally favors **precision over recall**. A future document whose fund name appears only in weak prose may not be discovered.
- Alias resolution is conservative. This avoids merging distinct share classes/sub-funds but may miss true aliases with substantially different spelling.
- Discovery relies on recognizable document semantics. A very different prospectus template may require learned entity classification or a human-review queue.
- A larger system should maintain a versioned fund/entity registry independent of any one ingestion run and reconcile new evidence into that registry.

---

# Phase 3 — fund segmentation

**Implementation:** `src/fund_rag/fund_segmentation.py`  
**Evaluation:** `notebooks/fund_segmentation_eval.ipynb`

## Approach

Segmentation consumes the already-validated registry and document classifications. It does **not** try to rediscover entity names.

For a multi-fund document it locates ownership transitions using:

- explicit `Name / Page` fund-list references;
- exact known fund headings;
- “Information concerning the sub-fund” markers;
- fund-specific annex headings.

A candidate heading inside a detected table is ignored so a table-of-contents/list row cannot become an accidental content boundary.

Segments use page/block start and end positions, with end positions treated as exclusive. Shared front matter can therefore remain `fund=None`, while a fund's section owns only its structural range.

## Evaluation result

The last validated prospectus run reported:

- **100% recall** on the representative manually checked fund boundaries;
- **44 continuous, non-overlapping prospectus segments**;
- correct standalone one-segment ownership for the single-fund documents;
- correct inspected transitions around page 38 and the late prospectus annexes.

The generic investor-rights document remains a general/no-fund segment rather than being manually assigned to individual funds.

## Limitations and possible improvements

- This is structural segmentation, so it depends on explicit headings/list references being present.
- General manager-level documents are currently represented as `fund=None`. A production ontology could explicitly associate a general document with all funds managed by the relevant entity.
- Repeated or unusual annex structures may need richer document-section semantics.
- A future production registry could store `fund`, `share class`, `manager`, `umbrella`, and relationship types separately rather than representing only fund ownership.

---

# Phase 4 — chunking

**Implementation:** `src/fund_rag/chunking.py`  
**Evaluation:** `notebooks/chunking_eval.ipynb`

## Approach

Chunking occurs **inside one structural segment at a time**. This is the main safety property of the chunker: overlap can never cross from one validated fund segment into another.

The chunker converts parsed page content into ordered units from:

- native text blocks;
- tables;
- chart-derived text.

Text blocks that substantially overlap extracted table/chart regions are excluded to avoid indexing the same visual content twice.

Heading signals from the ingestion layout metadata are used to keep sections readable and to populate a `section` label where possible.

Current defaults:

```text
max_words    = 300
min_words    = 80
overlap_words = 40
```

Each chunk preserves:

```text
chunk_id
document
fund
segment_type
segment_index
section
page_start
page_end
source_types
text
```

## Evaluation result

The last validated run generated **1,654 chunks**:

| Chunk statistic | Value |
|---|---:|
| Mean words | ~201 |
| Median words | ~222 |
| 90th percentile | 295 |
| 95th percentile | 300 |
| Maximum | 300 |
| Chunks above configured max | 0 |
| Chunks assigned to multiple segments | 0 |

Twenty-six chunks were below 50 words. Those occur at structural boundaries and were retained rather than forcibly merged across a meaningful boundary.

The notebook also inspects the known fund transition around page 38 and the 500+ annex region to confirm that chunks retain the expected fund ownership.

## Limitations and possible improvements

- Word-count chunking is simple and inspectable but does not directly optimize a transformer token budget.
- The current parameters were validated on this corpus. A larger system should tune chunking against retrieval/evidence recall rather than assuming one size fits every document type.
- Tables are serialized as text rows. More complex relational questions might benefit from separate table-aware retrieval.
- Chunk overlap increases index size. It is acceptable at this scale but should be evaluated against marginal recall in a much larger corpus.

---

# Phase 5 — indexing and retrieval

**Implementation:** `src/fund_rag/retrieval.py`  
**Evaluation:** `notebooks/retrieval_eval.ipynb`

## Retrieval design

### 1. Resolve fund names first

Known canonical names and aliases are matched against the query. Overlapping names use longest-match resolution so, for example, `K&H egészség 2` is not silently reduced to `K&H egészség`.

Fund routing is a hard candidate filter:

- named fund → only chunks owned by that fund;
- no named fund → only general/no-fund chunks.

This strongly protects against cross-fund contamination.

### 2. Dense retrieval with BGE-M3

The production embedding model is:

```text
BAAI/bge-m3
```

It was selected specifically for the English/Bulgarian cross-language requirement.

The dense vectors are normalized and stored in an inner-product FAISS index. The index metadata contains:

- index version;
- embedding model name;
- passage prefix configuration;
- chunk count;
- chunk fingerprint.

If those values still match, startup loads the existing FAISS index instead of re-embedding all 1,654 chunks.

### 3. Language-aware BM25

BM25 is built from the saved chunk text at startup. It adds lexical precision when the filtered candidate pool is monolingual.

### 4. `auto` policy

The live retriever uses:

```text
mixed EN/BG candidate pool -> semantic BGE-M3 retrieval
monolingual candidate pool -> dense + BM25 hybrid retrieval
```

This is intentional. In experiments, forcing lexical matching into cross-language questions could hurt retrieval because English query tokens cannot directly match Bulgarian evidence.

## Model-selection evidence

The controlled embedding-model comparison showed why BGE-M3 replaced multilingual E5:

| Case | multilingual-e5-base rank | BGE-M3 rank |
|---|---:|---:|
| EN → BG transfer | 49 | 2 |
| EN → BG rights | 31 | 4 |
| BG → BG transfer | 1 | 1 |
| BG → BG rights | 2 | 4 |
| EN → EN K&H objective | 1 | 1 |
| BG → EN K&H objective | 1 | 1 |

BGE-M3 fixed the most important cross-language failure while preserving top-5 performance in the other directions.

## Retrieval evaluation

On the current nine-case evidence benchmark, `auto` achieved:

| Metric | Result |
|---|---:|
| Source recall@5 | **100%** |
| Mean evidence coverage@5 | **94.44%** |
| Full-evidence recall@5 | **88.89%** |
| MRR | **0.8148** |
| Full-evidence recall in diversified top-50 candidates | **100%** |

The remaining top-5 weakness is the `enhanced_esg_risk` case, where only half of the expected evidence groups appeared in the first five chunks.

## Why the reranker is not in production

A multilingual BGE cross-encoder reranker was evaluated over retrieved candidates.

Observed comparison:

```text
mean evidence coverage@5: 0.9444 -> 0.9444
MRR:                     0.8148 -> 1.0000
mean added latency:      ~16.39 seconds/query
```

The reranker improved ordering of evidence that was already present but **did not recover additional required evidence**. For this small assistant the latency/cost trade-off was not justified, so reranking remains an optional evaluation experiment rather than part of the live path.

## Limitations and possible improvements

- A query that does not name a fund intentionally searches only general/no-fund chunks. This maximizes isolation but means vague fund-specific questions should be clarified rather than globally searched.
- General documents associated with many funds are not yet injected into a named-fund route. A richer `scope/related_funds` layer could support that safely.
- The ESG benchmark shows that top-5 evidence coverage is not perfect. Possible improvements include query decomposition, a slightly larger evidence candidate set, or selective reranking only for broad multi-evidence questions.
- FAISS `IndexFlatIP` is ideal at this scale. Millions of chunks would require an approximate index/vector database and incremental version management.

---

# Phase 6 — grounded answer generation

**Implementation:** `src/fund_rag/generator.py`  
**Evaluation:** `notebooks/generator_eval.ipynb`  
**Interactive interface:** `notebooks/assistant.ipynb`

## Responsibility split

The generator is intentionally **not** responsible for finding arbitrary corpus content.

Suppose retrieval returns:

```text
E1 -> unrelated securities-lending text
E2 -> "Fund units may be transferred freely ..."
E3 -> collateral-reuse text
E4 -> ...
```

Gemini sees those candidate chunks and decides which supplied evidence directly supports its answer. It may return:

```json
{
  "status": "supported",
  "answer": "Yes ...",
  "evidence_ids": ["E2"]
}
```

Python then validates that `E2` was genuinely supplied and maps it back to the original chunk metadata.

This creates a useful separation:

```text
Retriever: Which chunks are plausible evidence?
Generator: Which supplied chunks actually support the answer, and how should the answer be expressed?
Python: Are the citations valid, and what real source metadata do they represent?
```

## Prompt augmentation

The current evidence budget is bounded by:

```text
max_chunks               = 6
max_chars_per_chunk      = 4500
max_total_evidence_chars = 22000
```

The benchmark prompts observed in the last run were roughly **7k–13k characters**, so the configured ceiling was not approached in normal examples.

Each evidence block contains:

```text
[E2]
Fund: ...
Document: ...
Page: ...
Section: ...
Text:
...
```

The prompt instructs the model to:

- use only supplied evidence;
- answer in the user's language;
- keep funds separate;
- return `not_found` when evidence is absent;
- identify genuine ambiguity;
- identify conflicting supplied information;
- cite only evidence IDs actually used.

## Structured generation

Gemini returns a Pydantic-constrained structure with:

```text
status: supported | not_found | ambiguous | conflicting
answer: natural-language answer
evidence_ids: [E1, E2, ...]
```

A supported answer must cite at least one supplied evidence item. A conflicting answer must cite at least two. Unknown/hallucinated evidence IDs are rejected before the result is returned.

For multi-fund questions, retrieval runs independently for each named fund and interleaves the per-fund rankings before the six-chunk prompt budget is applied. This prevents the first fund in the question from consuming the whole generation context.

## API/provider considerations

This project was focused on the retrieval, grounding, attribution, multilingual evidence handling and evaluation of the different phases. 
Gemini service availability is an external dependency. During evaluation a temporary `503 UNAVAILABLE` due to high model demand was observed. The evaluation notebook now records provider failures separately rather than treating a transient HTTP failure as an incorrect RAG answer.

## Limitations and possible improvements

- Add explicit retry/backoff/circuit-breaking around transient provider errors in a production API service.
- Cache identical generation requests when appropriate.
- Track token usage, latency, and provider cost.
- Expand answer-quality evaluation beyond keyword/source checks to a larger manually reviewed set.
- Consider query decomposition for questions requiring many separate pieces of evidence; simply increasing context size is not always the best solution.
- A production service should read the API key from a secrets manager rather than an interactive notebook prompt.

---

# Source attribution and hallucination control

The strongest grounding property in this implementation is that **the LLM never authors source metadata**.

```text
retrieved chunk
    -> internal evidence ID E3
    -> Gemini cites E3
    -> Python validates E3
    -> Python reads E3.document / page / section / fund
```

The model can still make a reasoning mistake about the content of a valid chunk, but it cannot fabricate `document_name.pdf page 12` and have that fabricated location accepted as a source.

This also makes citation validity automatically testable.

---

# Multilingual behavior

The corpus contains Bulgarian and English evidence, and questions can be asked in either language.

Multilingual support is provided at several independent layers:

1. **Parsing:** native Unicode text is preserved.
2. **Fund resolution:** names/aliases are normalized without translating them.
3. **Dense retrieval:** BGE-M3 supports cross-language semantic matching.
4. **Hybrid policy:** BM25 is used only where lexical matching is appropriate.
5. **Generation:** the prompt requires an answer in the same language as the question.

This separation is important. Translation is not required as an ingestion step, and translating the entire corpus would introduce additional cost and another source of semantic drift.

---

# Evaluation strategy

The project deliberately does not use one aggregate “RAG accuracy” number. Each phase has a different contract:

| Phase | Main question |
|---|---|
| Ingestion | Did we preserve source information faithfully? |
| Fund discovery | Did we identify the right fund entities without false positives? |
| Segmentation | Did we assign document ranges to the correct fund? |
| Chunking | Are chunks bounded, readable, and fund-safe? |
| Retrieval | Did the top candidates contain the required source/evidence? |
| Generation | Did the model answer from supplied evidence and cite valid sources? |

This makes failures actionable. For example, the generator should not be blamed for missing ESG evidence if retrieval never supplied that evidence.

## Current evaluation summary

| Phase | Last validated result |
|---|---|
| Ingestion | 100% on current source-coverage/text/table/chart/image checks |
| Fund discovery | 41/41 funds, 100% precision/recall/F1 |
| Segmentation | 100% representative boundary recall; 44 continuous prospectus segments |
| Chunking | 1,654 chunks; 0 above max size; no cross-segment assignment |
| Retrieval `auto` | 100% source recall@5; 94.44% evidence coverage@5; 0.8148 MRR |
| Candidate retrieval | 100% full-evidence recall@50 |
| Reranker | no top-5 coverage gain; ~16.39 s/query added latency |
| Generation | structured grounding/citation checks implemented; provider failures isolated from quality metrics |

These are **development-benchmark results on the supplied corpus**, not a held-out estimate of performance on arbitrary investment documents.

---

# Interactive assistant

After the indexing notebooks have generated their artifacts, open:

```text
notebooks/assistant.ipynb
```

Run the setup cell once, then use only:

```python
ask("Can I transfer my fund units to another person?")
```

or:

```python
ask("Мога ли да прехвърля дяловете си на друго лице?")
```

The notebook prints the natural-language answer followed by the assignment-compatible JSON response.

---

# Reproducibility and artifact lifecycle

Generated artifacts are intentionally kept outside source control. The pipeline can recreate them in order.

```text
artifacts/ingestion/*.json
artifacts/fund_discovery/registry.json
artifacts/fund_discovery/classifications.json
artifacts/fund_discovery/candidates.json
artifacts/fund_segmentation/segments.json
artifacts/chunks/chunks.json
artifacts/retrieval_bge_m3/semantic.faiss
artifacts/retrieval_bge_m3/semantic_index.json
```

The retrieval index is protected by a chunk/model fingerprint, so changing the indexed chunks or embedding configuration invalidates the saved index and forces a rebuild.

---

# Production evolution

The assignment asks for a simple solution, so several enterprise features are intentionally not implemented. A larger production system would evolve each layer independently.

## Larger document collection

- Store document IDs, source versions, hashes, ingestion timestamps, and parser versions in persistent metadata.
- Perform incremental ingestion/indexing only for added or changed documents.
- Replace flat FAISS with an approximate/vector-database index suited to the expected scale and update pattern.
- Keep lexical and semantic indexes version-aligned with the same chunk set.
- Move model loading into long-lived services rather than notebook processes.


## Entity and ownership model

The current fund registry is intentionally lightweight. At scale, use stable IDs and explicit relations for:

- umbrella fund;
- sub-fund;
- share class;
- asset manager;
- feeder/master relationship;
- general manager-level documents;
- related documents and versions.

This would solve the current limitation where general/no-fund documents are not automatically included in a named-fund route.

## Retrieval

- Expand the labeled retrieval benchmark before changing models.
- Use query decomposition for multi-part questions.
- Consider selective reranking only when candidate uncertainty warrants the latency.
- Add intent-aware `k` rather than a fixed top-k for every query.
- Evaluate table-specific retrieval if more questions depend on complex tables.

# Known limitations

The most important current limitations are explicit rather than hidden:

1. **No OCR for scanned/image-only PDFs.**
2. **Visual understanding is limited to extracted images plus targeted chart-to-table conversion.**
3. **Fund discovery favors explicit structural evidence and may miss unusually formatted fund names.**
4. **General/no-fund documents are not automatically added to named-fund routes.**
5. **Generic questions that omit a fund intentionally search only general chunks; the system does not guess a fund.**
6. **Top-5 evidence recall is strong but not perfect; the ESG case currently exposes one multi-evidence weakness.**
7. **Reranking is omitted from the live path because measured coverage did not improve enough to justify latency.**
8. **Generation depends on an external Gemini API and therefore on provider availability/rate limits.**
9. **The evaluation set is small and development-oriented rather than held out.**

These are reasonable trade-offs for the assignment's requested scope and are preferable to hiding them behind more infrastructure.

---

# Automated tests

The test suite is organized by RAG phase:

```text
tests/
├── test_ingestion.py
├── test_fund_discovery.py
├── test_fund_segmentation.py
├── test_chunking.py
├── test_retrieval.py
├── test_generation.py
└── test_pipeline.py

---
```
## Example interactions

### Example 1 — unsupported question

```python
ask("What is the weather in Sofia?")


```json
{
  "answer": "I am sorry, but the provided evidence does not contain information about the weather in Sofia.",
  "funds": [],
  "sources": []
}
```

This demonstrates the assistant's `not_found` behavior: it does not answer from outside knowledge when the requested information is absent from the supplied fund documents.

### Example 2 — fund-specific question

```python
ask("What is the investment objective of K&H egészség?")
```

```json
{
  "answer": "The K&H egészség sub-fund has two investment objectives (before taxes and charges):\n1. To repay at maturity at least 100% of the initial subscription price (specifically 10,000 HUF per share, providing capital protection).\n2. To generate a potential capital gain or return, through an investment in swaps, that is contingent on the performance of a basket of 30 shares of companies in the pharmaceutical and healthcare industries.",
  "funds": [
    "K&H egészség"
  ],
  "sources": [
    {
      "document": "FU_BF065_EN.pdf",
      "page": 40,
      "section": "Investment objectives and strategy"
    },
    {
      "document": "FU_BF065_EN.pdf",
      "page": 38,
      "section": "Description of the sub-fund's object"
    }
  ]
}
```

This demonstrates a grounded fund-specific answer assembled from multiple retrieved source locations while preserving document, page, section, and fund metadata.