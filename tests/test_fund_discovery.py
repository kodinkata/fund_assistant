from __future__ import annotations

from fund_rag.fund_discovery import (
    _same_entity,
    build_registry,
    classify_documents,
    discover,
    extract_seed_candidates,
)


def test_discovers_fund_list_but_rejects_unrelated_name_role_table(multifund_document):
    candidates = extract_seed_candidates([multifund_document])
    names = {candidate["name"] for candidate in candidates}

    assert "Cedar Global Fund" in names
    assert "Oak Income Fund" in names
    assert "Alice Example" not in names
    assert "Bob Example" not in names


def test_similar_fund_names_remain_distinct_while_real_umbrella_alias_can_merge():
    assert not _same_entity("K&H egészség", "K&H egészség 2")
    assert _same_entity(
        "Optimum Fund World Selection 100-1 Advanced",
        "World Selection 100-1 Advanced",
    )

    candidates = [
        {"name": "K&H egészség", "document": "a.pdf", "page": 1, "method": "relation", "evidence": ""},
        {"name": "K&H egészség 2", "document": "a.pdf", "page": 2, "method": "relation", "evidence": ""},
    ]
    registry = build_registry(candidates)
    assert [entry["canonical_name"] for entry in registry] == ["K&H egészség", "K&H egészség 2"]


def test_document_classification_covers_no_single_and_multi_fund(
    multifund_document,
    rights_document,
    single_fund_document,
):
    documents = [multifund_document, rights_document, single_fund_document]
    registry, classifications, _ = discover(documents)
    by_name = {row["document"]: row for row in classifications}

    registry_names = {entry["canonical_name"] for entry in registry}
    assert "Cedar Global Fund" in registry_names
    assert "Oak Income Fund" in registry_names

    assert by_name["prospectus.pdf"]["document_type"] == "multi_fund"
    assert by_name["investor_rights.pdf"]["document_type"] == "no_fund"
    assert by_name["cedar_kid.pdf"]["document_type"] == "single_fund"
    assert by_name["cedar_kid.pdf"]["primary_fund"] == "Cedar Global Fund"
