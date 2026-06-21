from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.append(str(Path(__file__).resolve().parents[2]))

from backend.app.services.tray_layout import standard_template_boxes
from ml.common import configure_utf8_stdout, iter_images, project_path
from ml.training.preview_tray_cell_annotations import make_previews


def project_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(project_path().resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def load_existing(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("images", payload) if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        raise SystemExit(f"Invalid annotation file: {path}")
    return {
        str(record.get("file") or record.get("image") or record.get("path")): record
        for record in records
        if record.get("file") or record.get("image") or record.get("path")
    }


def make_records(
    image_dir: Path,
    existing_annotations: Path,
    default_label: str | None,
) -> list[dict[str, Any]]:
    existing = load_existing(existing_annotations)
    records: list[dict[str, Any]] = []

    for image_path in iter_images(image_dir):
        source = project_relative(image_path)
        if source in existing and isinstance(existing[source].get("cells"), list):
            records.append(existing[source])
            continue

        cells = []
        for box in standard_template_boxes(image_path):
            cell: dict[str, Any] = {
                "cell_id": box["cell_id"],
                "box": list(box["box"]),
            }
            if default_label:
                cell["label"] = default_label
            cells.append(cell)

        records.append(
            {
                "file": source,
                "reviewed": False,
                "annotation_source": "fixed_single_tray_template",
                "needs_label_review": True,
                "cells": cells,
            }
        )
    return records


def main() -> None:
    configure_utf8_stdout()
    parser = argparse.ArgumentParser(
        description="Seed five fixed tray-cell boxes for the single competition tray."
    )
    parser.add_argument("--image-dir", default="data/raw/tray_with_food")
    parser.add_argument("--existing-annotations", default="data/annotations/tray_cells.json")
    parser.add_argument("--output", default="data/annotations/tray_cells_fixed_template.json")
    parser.add_argument("--preview-output", default="data/generated/annotation_previews_fixed_template")
    parser.add_argument(
        "--default-label",
        default=None,
        help="Optional label to put on every cell. Leave empty for manual food labels.",
    )
    parser.add_argument("--max-preview-images", type=int, default=None)
    args = parser.parse_args()

    image_dir = project_path(args.image_dir)
    existing_annotations = project_path(args.existing_annotations)
    output_path = project_path(args.output)
    preview_output = project_path(args.preview_output)

    if not image_dir.exists():
        raise SystemExit(f"Missing image directory: {image_dir}")
    if args.max_preview_images is not None and args.max_preview_images < 1:
        raise SystemExit("--max-preview-images must be >= 1")

    records = make_records(
        image_dir=image_dir,
        existing_annotations=existing_annotations,
        default_label=args.default_label,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps({"images": records}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    make_previews(records, output_path, preview_output, max_images=args.max_preview_images)
    print(f"images: {len(records)}")
    print(f"annotations: {output_path}")
    print(f"previews: {preview_output}")


if __name__ == "__main__":
    main()
