from __future__ import annotations

from fund_rag.fund_discovery import classify_documents
from fund_rag.fund_segmentation import find_fund_anchors, segment_document, segment_documents


def test_multifund_document_segments_at_real_fund_boundaries(multifund_document, registry):
    classifications = classify_documents([multifund_document], registry)
    classification = classifications[0]
    # This fixture's registry has no evidence entries, so force the known multi-fund
    # document type; segmentation itself is what this test is validating.
    classification = {**classification, "document_type": "multi_fund", "primary_fund": None}

    anchors = find_fund_anchors(multifund_document, registry)
    assert [(a["page"], a["fund"]) for a in anchors] == [
        (2, "Cedar Global Fund"),
        (3, "Oak Income Fund"),
    ]

    segments = segment_document(multifund_document, registry, classification)

    assert segments[0] == {
        "document": "prospectus.pdf",
        "fund": None,
        "segment_type": "shared",
        "start_page": 1,
        "start_block": 0,
        "end_page": 2,
        "end_block": 0,
    }
    assert segments[1]["fund"] == "Cedar Global Fund"
    assert (segments[1]["start_page"], segments[1]["end_page"]) == (2, 3)
    assert segments[2]["fund"] == "Oak Income Fund"
    assert segments[2]["end_page"] is None


def test_single_and_general_documents_keep_whole_document_scope(
    rights_document,
    single_fund_document,
    registry,
):
    classifications = [
        {
            "document": "investor_rights.pdf",
            "document_type": "no_fund",
            "primary_fund": None,
            "mentioned_funds": [],
        },
        {
            "document": "cedar_kid.pdf",
            "document_type": "single_fund",
            "primary_fund": "Cedar Global Fund",
            "mentioned_funds": ["Cedar Global Fund"],
        },
    ]

    segments = segment_documents(
        [rights_document, single_fund_document],
        registry,
        classifications,
    )
    by_doc = {segment["document"]: segment for segment in segments}

    assert by_doc["investor_rights.pdf"]["scope"] == "general"
    assert by_doc["investor_rights.pdf"]["fund"] is None
    assert by_doc["cedar_kid.pdf"]["scope"] == "fund"
    assert by_doc["cedar_kid.pdf"]["fund"] == "Cedar Global Fund"
