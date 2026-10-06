from __future__ import annotations

import json
from pathlib import Path

import pymupdf
import pytest

from fund_rag.ingestion import extract_document, extract_pdf, extract_txt, run_ingestion


def test_extract_txt_preserves_text_and_rag_representation(tmp_path):
    source = tmp_path / "rights.txt"
    source.write_text("Transfer of units\nUnits may be transferred freely.", encoding="utf-8")

    parsed = extract_txt(source)

    assert parsed["document"] == "rights.txt"
    assert len(parsed["pages"]) == 1
    page = parsed["pages"][0]
    assert page["text"] == page["text_for_rag"]
    assert "transferred freely" in page["text"]
    assert page["text_blocks"][0]["text"] == page["text"]
    assert page["tables"] == []
    assert page["charts"] == []


def test_extract_pdf_uses_real_pymupdf_and_preserves_page_layout(tmp_path):
    source = tmp_path / "sample.pdf"
    doc = pymupdf.open()
    first = doc.new_page()
    first.insert_text((72, 72), "Transfer of units", fontsize=16)
    first.insert_text((72, 110), "Units may be transferred freely.", fontsize=10)
    second = doc.new_page()
    second.insert_text((72, 72), "Second page evidence", fontsize=11)
    doc.save(source)
    doc.close()

    parsed = extract_pdf(source, tmp_path / "artifacts")

    assert parsed["document"] == "sample.pdf"
    assert [p["page"] for p in parsed["pages"]] == [1, 2]
    assert "Transfer of units" in parsed["pages"][0]["text"]
    assert "Units may be transferred freely" in parsed["pages"][0]["text_for_rag"]
    assert parsed["pages"][0]["text_blocks"]
    assert all("bbox" in b and "font_size" in b for b in parsed["pages"][0]["text_blocks"])


def test_sample_rights_pdf_contains_expected_real_evidence(tmp_path):
    root = Path(__file__).resolve().parents[1]
    source = root / "data" / "documents" / "11074.pdf"
    assert source.exists(), "The assignment PDF should be included in data/documents."

    parsed = extract_pdf(source, tmp_path / "artifacts")
    text = "\n".join(page["text"] for page in parsed["pages"])

    assert "Прехвърляне на дялове" in text
    assert "могат да се прехвърлят свободно" in text
    assert parsed["pages"][0]["text_blocks"]


def test_run_ingestion_writes_reusable_json_artifacts(tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "parsed"
    input_dir.mkdir()
    (input_dir / "a.txt").write_text("Alpha evidence", encoding="utf-8")
    (input_dir / "b.txt").write_text("Beta evidence", encoding="utf-8")

    documents = run_ingestion(input_dir, output_dir)

    assert [d["document"] for d in documents] == ["a.txt", "b.txt"]
    for name in ("a", "b"):
        artifact = output_dir / f"{name}.json"
        assert artifact.exists()
        saved = json.loads(artifact.read_text(encoding="utf-8"))
        assert saved["document"] == f"{name}.txt"
        assert saved["pages"][0]["text_for_rag"]


def test_extract_document_rejects_unsupported_type(tmp_path):
    source = tmp_path / "bad.docx"
    source.write_bytes(b"not supported")

    with pytest.raises(ValueError, match="Unsupported file type"):
        extract_document(source, tmp_path)
