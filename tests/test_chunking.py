from __future__ import annotations

from fund_rag.chunking import _page_units, chunk_documents, chunk_segment


def test_chunking_never_crosses_fund_segment_boundaries(multifund_document):
    segments = [
        {
            "document": "prospectus.pdf",
            "fund": "Cedar Global Fund",
            "segment_type": "fund_section",
            "start_page": 2,
            "start_block": 0,
            "end_page": 3,
            "end_block": 0,
        },
        {
            "document": "prospectus.pdf",
            "fund": "Oak Income Fund",
            "segment_type": "fund_section",
            "start_page": 3,
            "start_block": 0,
            "end_page": None,
            "end_block": None,
        },
    ]

    chunks = chunk_documents([multifund_document], segments, max_words=40, min_words=5, overlap_words=5)
    cedar = [c for c in chunks if c["fund"] == "Cedar Global Fund"]
    oak = [c for c in chunks if c["fund"] == "Oak Income Fund"]

    assert cedar and oak
    assert all("Oak seeks regular income" not in c["text"] for c in cedar)
    assert all("Cedar invests globally" not in c["text"] for c in oak)
    assert all(c["page_end"] <= 2 for c in cedar)
    assert all(c["page_start"] >= 3 for c in oak)


def test_structured_table_replaces_overlapping_text_in_page_units():
    page = {
        "page": 1,
        "text_blocks": [
            {
                "text": "Fee 0.80% duplicated from table",
                "bbox": [10, 100, 300, 120],
                "font_size": 10,
                "bold": False,
            },
            {
                "text": "Narrative text outside table",
                "bbox": [10, 20, 300, 40],
                "font_size": 10,
                "bold": False,
            },
        ],
        "tables": [{"bbox": [0, 90, 400, 160], "rows": [["Fee", "0.80%"]]}],
        "charts": [],
    }

    units = _page_units(page)
    texts = [u["text"] for u in units]

    assert "Narrative text outside table" in texts
    assert "Fee | 0.80%" in texts
    assert "Fee 0.80% duplicated from table" not in texts
    assert {u["source_type"] for u in units} == {"text", "table"}


def test_chunk_size_is_bounded_and_overlap_preserves_context(rights_document):
    # Inflate the body to force more than one chunk.
    body = " ".join(["transfer evidence"] * 80)
    rights_document["pages"][0]["text_blocks"].append({
        "text": body,
        "bbox": [0, 120, 500, 200],
        "font_size": 10,
        "bold": False,
    })
    segment = {
        "document": "investor_rights.pdf",
        "fund": None,
        "segment_type": "no_fund",
        "start_page": 1,
        "start_block": 0,
        "end_page": None,
        "end_block": None,
    }

    chunks = chunk_segment(
        rights_document,
        segment,
        segment_index=0,
        max_words=50,
        min_words=10,
        overlap_words=10,
    )

    assert len(chunks) > 1
    assert all(c["word_count"] <= 50 for c in chunks)
    assert all(c["document"] == "investor_rights.pdf" for c in chunks)
    assert any(c["section"] == "Transfer of units" for c in chunks)
