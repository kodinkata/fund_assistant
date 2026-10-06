from pathlib import Path
import json

import pymupdf
from PIL import Image
from transformers import AutoProcessor, Pix2StructForConditionalGeneration


# Load DePlot lazily because it is only needed when we find a chart.
processor = None
model = None


def read_chart(path: Path):
    """Convert a chart image into rows using DePlot."""
    global processor, model

    if model is None:
        processor = AutoProcessor.from_pretrained("google/deplot")
        model = Pix2StructForConditionalGeneration.from_pretrained(
            "google/deplot"
        )

    inputs = processor(
        images=Image.open(path).convert("RGB"),
        text="Generate the underlying data table of this chart.",
        return_tensors="pt",
    )

    output = model.generate(**inputs, max_new_tokens=512)
    text = processor.decode(output[0], skip_special_tokens=True)

    return [
        [cell.strip() for cell in line.split("|")]
        for line in text.replace("<0x0A>", "\n").splitlines()
        if "|" in line and not line.startswith("TITLE")
    ]


def table_rows(table):
    """Clean rows returned by PyMuPDF table extraction."""
    return [
        ["" if cell is None else " ".join(str(cell).split()) for cell in row]
        for row in table.extract()
        if any(cell for cell in row)
    ]


def overlap(first, second):
    """Fraction of first rectangle covered by second."""
    intersection = first & second

    if intersection.is_empty or first.get_area() == 0:
        return 0

    return intersection.get_area() / first.get_area()

def extract_text_blocks(page):
    """Preserve text layout features useful for headings / section detection."""
    result = []

    for block in page.get_text("dict", sort=True)["blocks"]:
        if block["type"] != 0:
            continue

        spans = [
            span
            for line in block["lines"]
            for span in line["spans"]
            if span["text"].strip()
        ]

        if not spans:
            continue

        text = " ".join(span["text"].strip() for span in spans)

        result.append({
            "text": text,
            "bbox": list(block["bbox"]),
            "font_size": max(span["size"] for span in spans),
            "bold": any(span["flags"] & 16 for span in spans),
        })

    return result
            
def extract_pdf(pdf_path: Path, output_dir: Path):
    pages = []
    saved_images = {}

    image_dir = output_dir / "images" / pdf_path.stem
    chart_dir = output_dir / "charts" / pdf_path.stem

    image_dir.mkdir(parents=True, exist_ok=True)
    chart_dir.mkdir(parents=True, exist_ok=True)

    with pymupdf.open(pdf_path) as pdf:

        for page_number, page in enumerate(pdf, start=1):

            # ---------------------------------------------------------
            # 1. Text
            # ---------------------------------------------------------
            text = page.get_text("text", sort=True).strip()
            text_blocks = extract_text_blocks(page)

            # ---------------------------------------------------------
            # 2. Tables
            # ---------------------------------------------------------
            tables = []

            for table in page.find_tables().tables:
                rows = table_rows(table)

                # One-row drawing grids are often chart geometry.
                if len(rows) > 1:
                    tables.append({
                        "bbox": list(table.bbox),
                        "rows": rows,
                    })

            # ---------------------------------------------------------
            # 3. Embedded images
            # ---------------------------------------------------------
            images = []

            for image in page.get_images(full=True):
                xref = image[0]

                if xref not in saved_images:
                    data = pdf.extract_image(xref)

                    path = image_dir / f"{xref}.{data['ext']}"
                    path.write_bytes(data["image"])

                    saved_images[xref] = str(path)

                images.append(saved_images[xref])

            # ---------------------------------------------------------
            # 4. Vector charts
            # ---------------------------------------------------------
            charts = []

            drawings = page.get_drawings()

            for index, region in enumerate(
                page.cluster_drawings(),
                start=1,
            ):
                region = pymupdf.Rect(region)

                # Ignore tiny drawing clusters.
                if region.width < 120 or region.height < 80:
                    continue

                # Count drawing primitives inside this cluster.
                primitives = sum(
                    len(drawing["items"])
                    for drawing in drawings
                    if pymupdf.Rect(
                        drawing["rect"]
                    ).intersects(region)
                )

                if primitives < 10:
                    continue

                # Avoid treating actual tables as charts.
                is_table = any(
                    overlap(
                        region,
                        pymupdf.Rect(table["bbox"]),
                    ) > 0.6
                    for table in tables
                )

                if is_table:
                    continue

                chart_path = (
                    chart_dir
                    / f"page_{page_number}_{index}.png"
                )

                page.get_pixmap(
                    matrix=pymupdf.Matrix(3, 3),
                    clip=region,
                    alpha=False,
                ).save(chart_path)

                rows = read_chart(chart_path)

                if len(rows) > 1:
                    charts.append({
                        "bbox": list(region),
                        "image": str(chart_path),
                        "rows": rows,
                    })

            # ---------------------------------------------------------
            # Text representation used later by RAG
            # ---------------------------------------------------------
            table_text = [
                "\n".join(" | ".join(row) for row in table["rows"])
                for table in tables
            ]

            chart_text = [
                "\n".join(" | ".join(row) for row in chart["rows"])
                for chart in charts
            ]

            text_for_rag = "\n\n".join(
                [text, *table_text, *chart_text]
            ).strip()

            pages.append({
                "page": page_number,
                "text_blocks": text_blocks,
                "text": text,
                "tables": tables,
                "charts": charts,
                "images": images,
                "text_for_rag": text_for_rag,
            })

    return {
        "document": pdf_path.name,
        "pages": pages,
    }


def extract_txt(path: Path):
    text = path.read_text(
        encoding="utf-8",
        errors="replace",
    )

    return {
        "document": path.name,
        "pages": [{
            "page": 1,
            "text": text,
            "text_blocks": [{
                "text": text,
                "bbox": None,
                "font_size": None,
                "bold": False,
            }],
            "tables": [],
            "charts": [],
            "images": [],
            "text_for_rag": text,
        }],
    }


def extract_document(path: Path, output_dir: Path):
    if path.suffix.lower() == ".pdf":
        return extract_pdf(path, output_dir)

    if path.suffix.lower() == ".txt":
        return extract_txt(path)

    raise ValueError(
        f"Unsupported file type: {path.suffix}"
    )


def run_ingestion(
    input_dir: Path | str,
    output_dir: Path | str,
):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(
        path
        for path in input_dir.iterdir()
        if path.suffix.lower() in {".pdf", ".txt"}
    )

    documents = []

    for path in files:
        print(f"Parsing {path.name}")

        document = extract_document(
            path,
            output_dir,
        )

        documents.append(document)

        json_path = output_dir / f"{path.stem}.json"

        json_path.write_text(
            json.dumps(
                document,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    return documents