"""Fund-aware segmentation before chunking."""

from __future__ import annotations

import re
import unicodedata


SPACE = re.compile(r"\s+")


def clean(value: str) -> str:
    return SPACE.sub(" ", value or "").strip(" \t\r\n:;,.\"'„“”")


def norm(value: str) -> str:
    return clean(unicodedata.normalize("NFKC", value or "")).casefold()


def _fund_match(text: str, registry: list[dict], exact=False) -> str | None:
    value = norm(text)
    matches = []

    for fund in registry:
        for name in {fund["canonical_name"], *fund.get("aliases", [])}:
            candidate = norm(name)
            if candidate and ((value == candidate) if exact else (candidate in value)):
                matches.append((len(candidate), fund["canonical_name"]))

    return max(matches)[1] if matches else None


def _inside_table(block: dict, page: dict) -> bool:
    """Do not treat names inside a table/list as section headings."""
    bbox = block.get("bbox")
    if not bbox:
        return False

    x0, y0, x1, y1 = bbox
    area = max(0, x1 - x0) * max(0, y1 - y0)
    if not area:
        return False

    for table in page.get("tables", []):
        tb = table.get("bbox") if isinstance(table, dict) else None
        if not tb:
            continue

        tx0, ty0, tx1, ty1 = tb
        intersection = (
            max(0, min(x1, tx1) - max(x0, tx0))
            * max(0, min(y1, ty1) - max(y0, ty0))
        )
        if intersection / area >= 0.5:
            return True

    return False


def _locate_fund(page: dict, fund: str, registry: list[dict]) -> int | None:
    marker = "information concerning the sub-fund"

    for i, block in enumerate(page.get("text_blocks", [])):
        if _inside_table(block, page):
            continue

        text = block.get("text", "")
        if _fund_match(text, registry, exact=True) == fund:
            return i

        if marker in norm(text) and _fund_match(text, registry) == fund:
            return i

    return None


def _list_anchors(document: dict, registry: list[dict]) -> list[dict]:
    """Use explicit Name/Page fund-list tables when available."""
    pages = {p["page"]: p for p in document.get("pages", [])}
    anchors = []

    for page in document.get("pages", []):
        for table in page.get("tables", []):
            rows = table.get("rows", []) if isinstance(table, dict) else []
            if len(rows) < 2:
                continue

            header = [norm(str(v or "")) for v in rows[0]]
            name_col = next(
                (i for i, h in enumerate(header)
                 if h in {"name", "fund", "fund name", "sub-fund", "subfund", "sub-fund name"}),
                None,
            )
            page_col = next(
                (i for i, h in enumerate(header)
                 if h in {"page", "страница", "стр"}),
                None,
            )
            if name_col is None or page_col is None:
                continue

            for row in rows[1:]:
                if not isinstance(row, list) or max(name_col, page_col) >= len(row):
                    continue

                fund = _fund_match(str(row[name_col] or ""), registry, exact=True)
                page_match = re.search(r"\d+", str(row[page_col] or ""))
                target = int(page_match.group()) if page_match else None

                if fund is None or target not in pages:
                    continue

                block = _locate_fund(pages[target], fund, registry)
                if block is not None:
                    anchors.append({
                        "document": document["document"],
                        "page": target,
                        "block_index": block,
                        "fund": fund,
                        "segment_type": "fund_section",
                        "source": "fund_list",
                    })

    return anchors


def _heading_anchors(document: dict, registry: list[dict]) -> list[dict]:
    """Find explicit fund headings and fund-specific annex headings."""
    marker = "information concerning the sub-fund"
    anchors = []

    for page in document.get("pages", []):
        for i, block in enumerate(page.get("text_blocks", [])):
            if _inside_table(block, page):
                continue

            text = clean(block.get("text", ""))
            value = norm(text)
            if not value:
                continue

            if value.startswith(("annex for ", "appendix for ")):
                fund = _fund_match(text, registry)
                kind, source = "annex", "annex_heading"
            else:
                fund = _fund_match(text, registry, exact=True)
                source = "exact_heading"
                if fund is None and marker in value:
                    fund = _fund_match(text, registry)
                    source = "subfund_marker"
                kind = "fund_section"

            if fund:
                anchors.append({
                    "document": document["document"],
                    "page": page["page"],
                    "block_index": i,
                    "fund": fund,
                    "segment_type": kind,
                    "source": source,
                })

    return anchors


def find_fund_anchors(document: dict, registry: list[dict]) -> list[dict]:
    """Return reliable places where fund ownership changes."""
    anchors = _list_anchors(document, registry) + _heading_anchors(document, registry)
    priority = {"annex_heading": 0, "fund_list": 0, "subfund_marker": 1, "exact_heading": 2}

    best = {}
    for anchor in anchors:
        key = (anchor["page"], anchor["fund"], anchor["segment_type"])
        current = best.get(key)

        if current is None or (
            priority[anchor["source"]], anchor["block_index"]
        ) < (
            priority[current["source"]], current["block_index"]
        ):
            best[key] = anchor

    ordered = sorted(best.values(), key=lambda x: (x["page"], x["block_index"]))

    # Ignore repeated headings while the same fund remains active.
    result = []
    for anchor in ordered:
        if result and result[-1]["fund"] == anchor["fund"]:
            continue
        result.append(anchor)

    return result


def segment_document(document: dict, registry: list[dict], classification: dict) -> list[dict]:
    """Create structural segments. End positions are exclusive."""
    pages = document.get("pages", [])
    if not pages:
        return []

    start = (pages[0]["page"], 0)
    kind = classification["document_type"]

    if kind != "multi_fund":
        return [{
            "document": document["document"],
            "fund": classification["primary_fund"] if kind == "single_fund" else None,
            "segment_type": kind,
            "start_page": start[0],
            "start_block": start[1],
            "end_page": None,
            "end_block": None,
        }]

    anchors = find_fund_anchors(document, registry)
    if not anchors:
        return [{
            "document": document["document"],
            "fund": None,
            "segment_type": "shared",
            "start_page": start[0],
            "start_block": start[1],
            "end_page": None,
            "end_block": None,
        }]

    segments = []
    first = anchors[0]

    if (first["page"], first["block_index"]) != start:
        segments.append({
            "document": document["document"],
            "fund": None,
            "segment_type": "shared",
            "start_page": start[0],
            "start_block": start[1],
            "end_page": first["page"],
            "end_block": first["block_index"],
        })

    for i, anchor in enumerate(anchors):
        next_anchor = anchors[i + 1] if i + 1 < len(anchors) else None
        segments.append({
            "document": document["document"],
            "fund": anchor["fund"],
            "segment_type": anchor["segment_type"],
            "start_page": anchor["page"],
            "start_block": anchor["block_index"],
            "end_page": next_anchor["page"] if next_anchor else None,
            "end_block": next_anchor["block_index"] if next_anchor else None,
        })

    return segments
    
def _with_scope(
    segment: dict,
    shared_fund_relations: dict[str, list[str]],
) -> dict:
    segment = dict(segment)

    if segment.get("fund"):
        segment["scope"] = "fund"
        segment["related_funds"] = []
    else:
        related = shared_fund_relations.get(
            segment["document"],
            [],
        )

        segment["scope"] = (
            "shared_funds"
            if related
            else "general"
        )
        segment["related_funds"] = list(related)

    return segment

def segment_documents(
    documents: list[dict],
    registry: list[dict],
    classifications: list[dict],
    shared_fund_relations: dict[str, list[str]] | None = None,
) -> list[dict]:
    by_doc = {
        c["document"]: c
        for c in classifications
    }

    shared_fund_relations = (
        shared_fund_relations or {}
    )

    return [
        _with_scope(
            segment,
            shared_fund_relations,
        )
        for document in documents
        for segment in segment_document(
            document,
            registry,
            by_doc[document["document"]],
        )
    ]