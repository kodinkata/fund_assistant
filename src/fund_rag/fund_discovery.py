"""Cold-start fund discovery before chunking.

Design:
1. Bootstrap only from high-precision evidence:
   - explicit "<name> is/е a sub-fund" relations
   - Product/Fund/Sub-fund name labels
   - fund-list tables with fund context
   - first-page quoted/prominent title followed by explicit fund/sub-fund context
2. Canonicalize conservatively (never collapse names just because they are similar).
3. Use the resulting registry to classify documents and detect multi-fund section starts.

No API LLM and no local language model required.
"""

from __future__ import annotations

import re
import statistics
import unicodedata

SPACE = re.compile(r"\s+")
FUND_TERM = re.compile(
    r"\b(?:sub[- ]?funds?|funds?|"
    r"подфонд(?:ът|а|и|ове)?|фонд(?:ът|а|и|ове)?)\b",
    re.I,
)
FUND_LIST = re.compile(
    r"\b(?:list|списък)\b.{0,100}\b(?:sub[- ]?funds?|funds?|"
    r"подфонд(?:ове|и)?|фонд(?:ове|и)?)\b",
    re.I | re.S,
)

RELATIONS = [
    re.compile(
        r"(?:^|[.!?]\s+|\n\n)\s*"
        r"(?P<name>[A-ZА-ЯЁ„“][^.!?\n]{4,180}?)\s+"
        r"(?:който\s+)?е\s+(?:подфонд|фонд)\b",
        re.I | re.M,
    ),
    re.compile(
        r"(?:^|[.!?]\s+|\n\n)\s*"
        r"(?P<name>[A-ZА-ЯЁ„“][^.!?\n]{4,180}?)\s+"
        r"(?:which\s+)?(?:is|constitutes|represents)\s+(?:an?\s+)?"
        r"(?:sub[- ]?fund|fund)\b",
        re.I | re.M,
    ),
]

LABEL = re.compile(
    r"^(?:product(?:\s+name)?|fund\s+name|sub[- ]?fund\s+name|"
    r"продукт|име\s+на\s+(?:фонда|подфонда)|"
    r"наименование\s+на\s+(?:фонда|подфонда))\s*:?\s*(?P<value>.*)$",
    re.I,
)

BAD_EXACT = {
    "summary", "резюме", "обобщение", "prospectus", "проспект",
    "selected strategy", "basic details", "introduction",
    "product", "продукт", "name", "име",
}


def clean(value: str) -> str:
    return SPACE.sub(" ", value or "").strip(" \t\r\n:;,.\"'„“”")


def norm(value: str) -> str:
    return clean(unicodedata.normalize("NFKC", value or "")).casefold()


def valid_name(value: str) -> bool:
    value = clean(value)
    if not (5 <= len(value) <= 180):
        return False
    if len(value.split()) < 2 or len(value.split()) > 24:
        return False
    if sum(ch.isalpha() for ch in value) < 4:
        return False
    if norm(value) in BAD_EXACT:
        return False
    if norm(value).startswith(("този продукт", "this product", "този фонд", "this fund")):
        return False
    if value.endswith((".", "!", "?")):
        return False
    return True


def _blocks(page: dict) -> list[dict]:
    """Normalize layout blocks from ingestion.py."""
    result = []
    for block in page.get("text_blocks", []):
        text = clean(block.get("text", ""))
        if not text:
            continue
        result.append({
            "text": text,
            "font_size": block.get("font_size"),
            "bold": bool(block.get("bold")),
        })
    return result


def _rows(table) -> list[list]:
    """Support the common table shapes produced by parsers."""
    if isinstance(table, list):
        return table if table and isinstance(table[0], list) else []
    if not isinstance(table, dict):
        return []
    for key in ("rows", "data", "cells"):
        value = table.get(key)
        if isinstance(value, list) and (not value or isinstance(value[0], list)):
            return value
    return []


def _name_column(header: list) -> int | None:
    for i, value in enumerate(header):
        h = norm(str(value or ""))
        if h in {"name", "име", "наименование", "fund", "sub-fund", "subfund"}:
            return i
        if "fund name" in h or "subfund name" in h or "sub-fund name" in h:
            return i
    return None


def _page_column(header: list) -> int | None:
    for i, value in enumerate(header):
        if norm(str(value or "")) in {"page", "страница", "стр"}:
            return i
    return None


def extract_seed_candidates(documents: list[dict]) -> list[dict]:
    """
    Cold-start extraction.

    Important: generic headings are NOT allowed to become entities merely because
    they are prominent. Every seed needs explicit fund semantics.
    """
    found = []

    def add(name, document, page, method, evidence=""):
        name = clean(name)
        if valid_name(name):
            found.append({
                "name": name,
                "document": document,
                "page": page,
                "method": method,
                "evidence": clean(evidence)[:300],
            })

    for document in documents:
        filename = document["document"]
        list_table_active = False
        previous_page = None

        for page in sorted(document.get("pages", []), key=lambda p: p["page"]):
            number = page["page"]
            text = page.get("text", "")
            blocks = _blocks(page)

            if previous_page is not None and number != previous_page + 1:
                list_table_active = False
            previous_page = number

            # A. Explicit linguistic relations.
            for pattern in RELATIONS:
                for match in pattern.finditer(text):
                    add(
                        match.group("name"), filename, number,
                        "relation", match.group(0)
                    )

            # B. Explicit labels: Product / Product Name / Fund Name / Sub-fund Name.
            block_texts = [b["text"] for b in blocks]
            for i, value in enumerate(block_texts):
                match = LABEL.match(value)
                if not match:
                    continue
                same_line = clean(match.group("value"))
                if same_line:
                    add(same_line, filename, number, "label", value)
                elif i + 1 < len(block_texts):
                    add(
                        block_texts[i + 1], filename, number,
                        "label_next", f"{value} | {block_texts[i + 1]}"
                    )

            # C. First-page title with explicit nearby fund semantics.
            # Layout supports the decision; it cannot create an entity by itself.
            if number == 1 and blocks:
                sizes = [
                    b["font_size"] for b in blocks
                    if isinstance(b.get("font_size"), (int, float)) and b["font_size"]
                ]
                body_size = statistics.median(sizes) if sizes else None

                for i, block in enumerate(blocks[:3]):
                    value = block["text"]
                    if not valid_name(value) or len(value.split()) > 16:
                        continue

                    nearby = " ".join(b["text"] for b in blocks[i:min(len(blocks), i + 4)])
                    prominent = bool(block["bold"]) or (
                        body_size is not None
                        and block.get("font_size") is not None
                        and block["font_size"] >= body_size * 1.25
                    )
                    quoted_first = i == 0 and text.lstrip().startswith(("„", '"'))

                    if prominent and quoted_first and FUND_TERM.search(nearby):
                        add(value, filename, number, "quoted_title", nearby)

            # D. Fund-list tables. A generic Name column is accepted only when
            # the page explicitly says it is a list of funds/sub-funds, or when
            # the table is a consecutive continuation of such a list.
            page_has_fund_list = bool(FUND_LIST.search(text))
            found_list_table = False

            for table in page.get("tables", []):
                rows = _rows(table)
                if len(rows) < 2:
                    continue

                header = rows[0]
                name_col = _name_column(header)
                page_col = _page_column(header)

                if name_col is None:
                    continue

                # A page can mention a list of funds while also containing unrelated
                # tables (for example a directors table headed ``Name``).  A generic
                # ``Name`` column therefore needs the stronger Name+Page list shape.
                # An explicitly fund-labelled column is already self-describing.
                header_label = norm(str(header[name_col] or ""))
                explicit_fund_header = (
                    "fund" in header_label
                    or "подфонд" in header_label
                    or "фонд" in header_label
                )
                list_shape = page_col is not None or explicit_fund_header
                continuation = list_table_active and page_col is not None

                if not list_shape or not (page_has_fund_list or continuation):
                    continue

                found_list_table = True
                for row in rows[1:]:
                    if isinstance(row, list) and name_col < len(row):
                        add(
                            str(row[name_col] or ""),
                            filename, number,
                            "fund_list_table",
                            f"header={rows[0]}",
                        )

            list_table_active = found_list_table and (
                page_has_fund_list or list_table_active
            )

    return found


def _same_entity(a: str, b: str) -> bool:
    """
    Conservative aliasing.

    Exact normalized names merge.
    Also allow a short umbrella prefix ending in 'Fund/фонд':
        'Optimum Fund World Selection 100-1 Advanced'
        -> 'World Selection 100-1 Advanced'

    Do NOT use generic fuzzy/substring matching here; names such as
    'K&H egészség' and 'K&H egészség 2' are distinct funds.
    """
    a, b = norm(a), norm(b)
    if a == b:
        return True

    short, long = sorted((a, b), key=len)
    if long.endswith(short):
        prefix = long[:-len(short)].strip()
        if (
            prefix
            and len(prefix.split()) <= 4
            and re.search(r"\b(?:fund|фонд)\b", prefix, re.I)
        ):
            return True

    return False


def build_registry(candidates: list[dict]) -> list[dict]:
    """Build a deterministic registry from high-precision seeds."""
    priority = {
        "fund_list_table": 0,
        "relation": 1,
        "quoted_title": 2,
        "label": 3,
        "label_next": 3,
    }
    groups = []

    for candidate in sorted(
        candidates,
        key=lambda c: (
            priority.get(c["method"], 9),
            c["document"],
            c["page"],
            norm(c["name"]),
        ),
    ):
        group = next(
            (
                g for g in groups
                if any(_same_entity(candidate["name"], alias) for alias in g["aliases"])
            ),
            None,
        )

        if group is None:
            group = {
                "canonical_name": candidate["name"],
                "aliases": [],
                "evidence": [],
            }
            groups.append(group)

        group["aliases"].append(candidate["name"])
        group["evidence"].append(candidate)

        # Prefer names directly enumerated in a fund list, then relation names.
        if candidate["method"] == "fund_list_table":
            group["canonical_name"] = candidate["name"]
        elif (
            candidate["method"] == "relation"
            and not any(e["method"] == "fund_list_table" for e in group["evidence"])
        ):
            group["canonical_name"] = candidate["name"]

    for group in groups:
        group["aliases"] = sorted(set(group["aliases"]), key=norm)

    return sorted(groups, key=lambda g: norm(g["canonical_name"]))


def registry_mentions(text: str, registry: list[dict]) -> list[str]:
    """Return canonical funds explicitly present in text."""
    haystack = norm(text)
    hits = []

    for fund in registry:
        aliases = sorted(fund["aliases"], key=len, reverse=True)
        if any(norm(alias) in haystack for alias in aliases):
            hits.append(fund["canonical_name"])

    return hits


def classify_documents(documents: list[dict], registry: list[dict]) -> list[dict]:
    """
    Classify documents after cold-start discovery:
      - multi_fund: explicit fund-list table / multiple fund section seeds
      - single_fund: exactly one registry entity dominates the document
      - no_fund: no registry fund found
    """
    output = []

    for document in documents:
        text = "\n".join(page.get("text", "") for page in document.get("pages", []))
        mentions = registry_mentions(text, registry)

        # Strong multi-fund evidence: this document contributed >1 list-table funds.
        list_funds = {
            fund["canonical_name"]
            for fund in registry
            if any(
                e["document"] == document["document"]
                and e["method"] == "fund_list_table"
                for e in fund["evidence"]
            )
        }

        if len(list_funds) > 1:
            kind = "multi_fund"
            primary = None
        elif len(mentions) == 1:
            kind = "single_fund"
            primary = mentions[0]
        elif len(mentions) == 0:
            kind = "no_fund"
            primary = None
        else:
            # Multiple mentions without a fund list need section analysis.
            kind = "multi_fund"
            primary = None

        output.append({
            "document": document["document"],
            "document_type": kind,
            "primary_fund": primary,
            "mentioned_funds": mentions,
        })

    return output

def discover(documents: list[dict]):
    candidates = extract_seed_candidates(documents)
    registry = build_registry(candidates)
    classifications = classify_documents(documents, registry)
    return registry, classifications, candidates
