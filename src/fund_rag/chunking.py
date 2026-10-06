"""Fund-aware chunking.

Chunks are created only inside an already validated fund segment.
A chunk can never cross from one fund segment into another.
"""

from __future__ import annotations

import re
import statistics


SPACE = re.compile(r"\s+")


def clean(text: str) -> str:
    return SPACE.sub(" ", text or "").strip()


def _words(text: str) -> list[str]:
    return clean(text).split()


def _overlap_fraction(first: list, second: list) -> float:
    if not first or not second or len(first) != 4 or len(second) != 4:
        return 0.0

    x0, y0, x1, y1 = first
    a0, b0, a1, b1 = second

    area = max(0, x1 - x0) * max(0, y1 - y0)
    if not area:
        return 0.0

    intersection = (
        max(0, min(x1, a1) - max(x0, a0))
        * max(0, min(y1, b1) - max(y0, b0))
    )
    return intersection / area


def _page_units(page: dict) -> list[dict]:
    """
    Convert one parsed page into ordered RAG units.

    Text inside detected tables/charts is skipped and replaced by the structured
    table/chart rows, avoiding duplicate content.
    """
    regions = [
        item.get("bbox")
        for item in [*page.get("tables", []), *page.get("charts", [])]
        if isinstance(item, dict) and item.get("bbox")
    ]

    units = []

    for block_index, block in enumerate(page.get("text_blocks", [])):
        bbox = block.get("bbox")

        if any(_overlap_fraction(bbox, region) >= 0.5 for region in regions):
            continue

        text = clean(block.get("text", ""))
        if not text:
            continue

        units.append({
            "text": text,
            "page": page["page"],
            "y": bbox[1] if bbox else 0,
            "source_type": "text",
            "block_index": block_index,
            "font_size": block.get("font_size"),
            "bold": bool(block.get("bold")),
        })

    for table in page.get("tables", []):
        rows = table.get("rows", []) if isinstance(table, dict) else []
        text = "\n".join(
            " | ".join(str(cell or "").strip() for cell in row)
            for row in rows
            if isinstance(row, list)
        ).strip()

        if text:
            bbox = table.get("bbox") or [0, 0, 0, 0]
            units.append({
                "text": text,
                "page": page["page"],
                "y": bbox[1],
                "source_type": "table",
                "block_index": None,
                "font_size": None,
                "bold": False,
            })

    for chart in page.get("charts", []):
        rows = chart.get("rows", []) if isinstance(chart, dict) else []
        text = "\n".join(
            " | ".join(str(cell or "").strip() for cell in row)
            for row in rows
            if isinstance(row, list)
        ).strip()

        if text:
            bbox = chart.get("bbox") or [0, 0, 0, 0]
            units.append({
                "text": text,
                "page": page["page"],
                "y": bbox[1],
                "source_type": "chart",
                "block_index": None,
                "font_size": None,
                "bold": False,
            })

    return sorted(
        units,
        key=lambda unit: (
            unit["y"],
            {"text": 0, "table": 1, "chart": 2}[unit["source_type"]],
        ),
    )


def _boundary_y(page: dict, block_index: int | None, default: float) -> float:
    if block_index is None:
        return default

    blocks = page.get("text_blocks", [])
    if not (0 <= block_index < len(blocks)):
        return default

    bbox = blocks[block_index].get("bbox")
    return bbox[1] if bbox else default


def segment_units(document: dict, segment: dict) -> list[dict]:
    """Return only the ordered content units that fall inside one segment."""
    pages = {page["page"]: page for page in document.get("pages", [])}
    output = []

    start_page = segment["start_page"]
    end_page = segment.get("end_page")

    for page_number in sorted(pages):
        if page_number < start_page:
            continue
        if end_page is not None and page_number > end_page:
            continue

        page = pages[page_number]
        start_y = float("-inf")
        end_y = float("inf")

        if page_number == start_page:
            start_y = _boundary_y(
                page,
                segment.get("start_block"),
                float("-inf"),
            )

        if end_page is not None and page_number == end_page:
            end_y = _boundary_y(
                page,
                segment.get("end_block"),
                float("inf"),
            )

        for unit in _page_units(page):
            if unit["y"] < start_y:
                continue

            # Segment ends are exclusive.
            if end_page is not None and page_number == end_page and unit["y"] >= end_y:
                continue

            output.append(unit)

    return output


_GENERIC_LABELS = {
    "name",
    "life",
    "office",
    "status",
    "currency",
    "legal form",
    "date of incorporation",
    "stock exchange listing",
}


def _body_font_size(units: list[dict]) -> float | None:
    sizes = [
        unit["font_size"]
        for unit in units
        if unit["source_type"] == "text"
        and isinstance(unit.get("font_size"), (int, float))
        and unit["font_size"] > 0
    ]
    return statistics.median(sizes) if sizes else None


def _is_heading(unit: dict, body_size: float | None) -> bool:
    if unit["source_type"] != "text":
        return False

    text = clean(unit["text"])
    words = _words(text)

    if not words or len(words) > 18:
        return False
    # Dates are metadata/content, not section headings.
    if re.fullmatch(r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}", text):
        return False
    
    if re.fullmatch(r"\d{1,2}\s+[A-Za-z]+\s+\d{4}", text):
        return False
    
    # Tiny numbered/form fragments are not useful section headings.
    if re.fullmatch(r"\(?\d+\)?[\s.:?-]*", text):
        return False
    
    if len(words) <= 2 and text.endswith("?"):
        return False
    if text.casefold().rstrip(":") in _GENERIC_LABELS:
        return False

    numbered = bool(re.match(r"^\d+(?:\.\d+)*\.?\s+\S", text))

    prominent = (
        body_size is not None
        and isinstance(unit.get("font_size"), (int, float))
        and unit["font_size"] >= body_size * 1.15
    )

    short_bold = unit.get("bold") and 2 <= len(words) <= 12

    return numbered or prominent or short_bold


def _split_unit(unit: dict, max_words: int) -> list[dict]:
    words = _words(unit["text"])
    if len(words) <= max_words:
        return [unit]

    parts = []
    for start in range(0, len(words), max_words):
        part = dict(unit)
        part["text"] = " ".join(words[start:start + max_words])
        part["continued"] = start > 0
        parts.append(part)

    return parts


def _tail_overlap(items: list[dict], overlap_words: int) -> list[dict]:
    if overlap_words <= 0 or not items:
        return []

    remaining = overlap_words
    carry = []

    for item in reversed(items):
        words = _words(item["text"])
        if not words:
            continue

        take = min(len(words), remaining)
        copied = dict(item)
        copied["text"] = " ".join(words[-take:])
        copied["overlap"] = True
        carry.append(copied)

        remaining -= take
        if remaining <= 0:
            break

    return list(reversed(carry))


def chunk_segment(
    document: dict,
    segment: dict,
    segment_index: int,
    max_words: int = 300,
    min_words: int = 80,
    overlap_words: int = 40,
) -> list[dict]:
    """Chunk one fund segment without ever crossing its borders."""
    units = segment_units(document, segment)
    if not units:
        return []

    body_size = _body_font_size(units)
    unit_limit = max_words - overlap_words
    
    expanded = [
        part
        for unit in units
        for part in _split_unit(unit, unit_limit)
    ]

    chunks = []
    buffer = []
    active_section = None
    chunk_number = 0

    def buffer_words():
        return sum(len(_words(item["text"])) for item in buffer)

    def flush(carry_overlap: bool):
        nonlocal buffer, chunk_number
        if not buffer:
            return

        text = "\n\n".join(item["text"] for item in buffer).strip()
        if not text:
            buffer = []
            return

        sections = []
        for item in buffer:
            section = item.get("section")
            if section and section not in sections:
                sections.append(section)

        chunks.append({
            "chunk_id": (
                f'{document["document"]}::segment_{segment_index:03d}'
                f'::chunk_{chunk_number:03d}'
            ),
            "document": document["document"],
            "fund": segment.get("fund"),
            "segment_type": segment.get("segment_type"),
            "segment_index": segment_index,
            "section": sections[-1] if sections else None,
            "page_start": min(item["page"] for item in buffer),
            "page_end": max(item["page"] for item in buffer),
            "word_count": len(_words(text)),
            "source_types": sorted({
                item["source_type"]
                for item in buffer
            }),
            "text": text,
        })

        chunk_number += 1
        buffer = _tail_overlap(buffer, overlap_words) if carry_overlap else []

    for unit in expanded:
        heading = _is_heading(unit, body_size)

        # Prefer starting meaningful new headings in a fresh chunk.
        if heading and buffer and buffer_words() >= min_words:
            flush(carry_overlap=False)

        if heading:
            active_section = clean(unit["text"])

        unit = dict(unit)
        unit["section"] = active_section

        unit_words = len(_words(unit["text"]))
        if buffer and buffer_words() + unit_words > max_words:
            flush(carry_overlap=True)

        buffer.append(unit)

    flush(carry_overlap=False)
    return chunks


def chunk_documents(
    documents: list[dict],
    segments: list[dict],
    max_words: int = 300,
    min_words: int = 80,
    overlap_words: int = 40,
) -> list[dict]:
    documents_by_name = {
        document["document"]: document
        for document in documents
    }

    segment_counts = {}
    chunks = []

    for segment in segments:
        document_name = segment["document"]
        segment_index = segment_counts.get(document_name, 0)
        segment_counts[document_name] = segment_index + 1

        chunks.extend(
            chunk_segment(
                documents_by_name[document_name],
                segment,
                segment_index,
                max_words=max_words,
                min_words=min_words,
                overlap_words=overlap_words,
            )
        )

    return chunks
