from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.append(str(Path(__file__).resolve().parents[2]))

from ml.common import configure_utf8_stdout, project_path
from ml.training.extract_tray_cell_crops import safe_name
from ml.training.prepare_tray_cell_detector_dataset import clamp_box, load_annotations, resolve_image_path


def require_pillow():
    try:
        from PIL import Image, ImageDraw, ImageFont, ImageOps
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: pillow. Install dependencies with: "
            "pip install -r backend/requirements.txt"
        ) from exc
    return Image, ImageDraw, ImageFont, ImageOps


def label_text(cell: dict[str, Any], index: int) -> str:
    cell_id = cell.get("cell_id", f"cell_{index:02d}")
    label = cell.get("label")
    if label:
        return f"{cell_id}: {label}"
    return str(cell_id)


def draw_labeled_box(draw: Any, box: tuple[int, int, int, int], text: str, line_width: int) -> None:
    x1, y1, x2, y2 = box
    color = (255, 64, 64)
    fill = (255, 64, 64)
    text_fill = (255, 255, 255)
    draw.rectangle(box, outline=color, width=line_width)

    text_bbox = draw.textbbox((x1, y1), text)
    text_w = text_bbox[2] - text_bbox[0]
    text_h = text_bbox[3] - text_bbox[1]
    pad = max(3, line_width)
    bg_y1 = max(0, y1 - text_h - pad * 2)
    bg_y2 = bg_y1 + text_h + pad * 2
    draw.rectangle((x1, bg_y1, x1 + text_w + pad * 2, bg_y2), fill=fill)
    draw.text((x1 + pad, bg_y1 + pad), text, fill=text_fill)


def make_previews(
    records: list[dict[str, Any]],
    annotation_path: Path,
    output_dir: Path,
    max_images: int | None,
) -> None:
    Image, ImageDraw, _, ImageOps = require_pillow()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, Any]] = []
    total_cells = 0
    selected_records = records[:max_images] if max_images is not None else records

    for image_index, record in enumerate(selected_records, start=1):
        raw_file = record.get("file") or record.get("image") or record.get("path")
        cells = record.get("cells")
        if not raw_file or not isinstance(cells, list):
            raise SystemExit("Each annotation record needs 'file' and a 'cells' list.")

        image_path = resolve_image_path(str(raw_file), annotation_path)
        if not image_path.exists():
            raise SystemExit(f"Annotated image not found: {image_path}")

        with Image.open(image_path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            width, height = image.size
            draw = ImageDraw.Draw(image)
            line_width = max(2, round(min(width, height) / 220))

            preview_cells: list[dict[str, Any]] = []
            for cell_index, cell in enumerate(cells, start=1):
                box = clamp_box(list(cell["box"]), width, height)
                draw_labeled_box(draw, box, label_text(cell, cell_index), line_width)
                preview_cells.append(
                    {
                        "cell_index": cell_index,
                        "cell_id": cell.get("cell_id", f"cell_{cell_index:02d}"),
                        "label": cell.get("label"),
                        "box": list(box),
                    }
                )
                total_cells += 1

            preview_path = output_dir / f"{image_index:04d}_{safe_name(image_path.stem)}.jpg"
            image.save(preview_path, quality=95)
            manifest.append(
                {
                    "source": str(image_path),
                    "preview": str(preview_path),
                    "width": width,
                    "height": height,
                    "cells": preview_cells,
                }
            )

    (output_dir / "manifest.json").write_text(
        json.dumps({"images": manifest}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {len(manifest)} preview images with {total_cells} boxes to {output_dir}")


def main() -> None:
    configure_utf8_stdout()
    parser = argparse.ArgumentParser(description="Draw tray-cell annotation previews.")
    parser.add_argument("--annotations", default="data/annotations/tray_cells.json")
    parser.add_argument("--output", default="data/generated/annotation_previews")
    parser.add_argument("--max-images", type=int, default=None)
    args = parser.parse_args()

    annotation_path = project_path(args.annotations)
    output_dir = project_path(args.output)
    if not annotation_path.exists():
        raise SystemExit(f"Annotation file not found: {annotation_path}")
    if args.max_images is not None and args.max_images < 1:
        raise SystemExit("--max-images must be >= 1")

    records = load_annotations(annotation_path)
    if not records:
        raise SystemExit("Annotation file has no images.")
    make_previews(records, annotation_path, output_dir, args.max_images)


if __name__ == "__main__":
    main()
