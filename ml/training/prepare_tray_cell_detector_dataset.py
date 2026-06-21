from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path
from typing import Any

sys.path.append(str(Path(__file__).resolve().parents[2]))

from ml.common import configure_utf8_stdout, project_path


def require_pillow():
    try:
        from PIL import Image, ImageOps
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: pillow. Install dependencies with: "
            "pip install -r backend/requirements.txt"
        ) from exc
    return Image, ImageOps


def load_annotations(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if isinstance(payload, dict):
        records = payload.get("images")
    else:
        records = payload
    if not isinstance(records, list):
        raise SystemExit("Annotation file must be a list or an object with an 'images' list.")
    return records


def resolve_image_path(raw_path: str, annotation_path: Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path

    candidates = [
        annotation_path.parent / path,
        project_path(path),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return project_path(path)


def clamp_box(box: list[float], width: int, height: int) -> tuple[int, int, int, int]:
    if len(box) != 4:
        raise ValueError("box must contain [x1, y1, x2, y2]")
    x1, y1, x2, y2 = [int(round(value)) for value in box]
    x1 = max(0, min(width, x1))
    x2 = max(0, min(width, x2))
    y1 = max(0, min(height, y1))
    y2 = max(0, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"invalid box after clamping: {[x1, y1, x2, y2]}")
    return x1, y1, x2, y2


def to_detector_line(box: tuple[int, int, int, int], image_size: tuple[int, int]) -> str:
    width, height = image_size
    x1, y1, x2, y2 = box
    cx = ((x1 + x2) / 2) / width
    cy = ((y1 + y2) / 2) / height
    bw = (x2 - x1) / width
    bh = (y2 - y1) / height
    return f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def prepare_output(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)
    for split in ["train", "val"]:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
    (output_dir / "metadata").mkdir(parents=True, exist_ok=True)


def write_data_yaml(output_dir: Path) -> None:
    content = "\n".join(
        [
            f"path: {output_dir.as_posix()}",
            "train: images/train",
            "val: images/val",
            "names:",
            "  0: tray_cell",
            "",
        ]
    )
    (output_dir / "data.yaml").write_text(content, encoding="utf-8")


def make_dataset(
    records: list[dict[str, Any]],
    annotation_path: Path,
    output_dir: Path,
    val_ratio: float,
    seed: int,
    reviewed_only: bool,
) -> None:
    Image, ImageOps = require_pillow()
    rng = random.Random(seed)
    if reviewed_only:
        records = [record for record in records if record.get("reviewed") is True]
    shuffled = list(records)
    if not shuffled:
        raise SystemExit(
            "No annotation records selected. Review auto-seeded annotations first, "
            "or rerun without --reviewed-only if you intentionally want to train from suggestions."
        )
    rng.shuffle(shuffled)
    val_count = int(round(len(shuffled) * val_ratio))

    metadata: list[dict[str, Any]] = []
    written_images = 0
    written_cells = 0

    for index, record in enumerate(shuffled):
        raw_file = record.get("file") or record.get("image") or record.get("path")
        cells = record.get("cells")
        if not raw_file or not isinstance(cells, list):
            raise SystemExit("Each annotation record needs 'file' and a 'cells' list.")

        image_path = resolve_image_path(str(raw_file), annotation_path)
        if not image_path.exists():
            raise SystemExit(f"Annotated image not found: {image_path}")

        split = "val" if index < val_count else "train"
        with Image.open(image_path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            width, height = image.size

            labels: list[str] = []
            cell_metadata: list[dict[str, Any]] = []
            for cell_index, cell in enumerate(cells, start=1):
                box = clamp_box(list(cell["box"]), width, height)
                labels.append(to_detector_line(box, (width, height)))
                cell_metadata.append(
                    {
                        "cell_index": cell_index,
                        "cell_id": cell.get("cell_id", f"cell_{cell_index:02d}"),
                        "box": list(box),
                        "label": cell.get("label"),
                    }
                )

            stem = f"tray_cell_{written_images:05d}"
            image.save(output_dir / "images" / split / f"{stem}.jpg", quality=95)
            (output_dir / "labels" / split / f"{stem}.txt").write_text(
                "\n".join(labels) + "\n",
                encoding="utf-8",
            )
            metadata.append(
                {
                    "source": str(image_path),
                    "split": split,
                    "image": f"images/{split}/{stem}.jpg",
                    "label_file": f"labels/{split}/{stem}.txt",
                    "cells": cell_metadata,
                }
            )
            written_images += 1
            written_cells += len(cell_metadata)

    write_data_yaml(output_dir)
    (output_dir / "metadata" / "cells.json").write_text(
        json.dumps({"images": metadata}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {written_images} images and {written_cells} tray_cell boxes to {output_dir}")
    print(f"detector config: {output_dir / 'data.yaml'}")


def main() -> None:
    configure_utf8_stdout()
    parser = argparse.ArgumentParser(description="Prepare detector data for tray-cell detection.")
    parser.add_argument("--annotations", default="data/annotations/tray_cells.json")
    parser.add_argument("--output", default="data/processed/tray_cells_detector")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--reviewed-only",
        action="store_true",
        help="Train only from records marked reviewed=true in the annotation JSON.",
    )
    args = parser.parse_args()

    annotation_path = project_path(args.annotations)
    output_dir = project_path(args.output)
    if not annotation_path.exists():
        raise SystemExit(
            f"Annotation file not found: {annotation_path}. "
            "Create it from real tray images before training tray_cell detector."
        )
    if not 0 <= args.val_ratio < 1:
        raise SystemExit("--val-ratio must be >= 0 and < 1")

    records = load_annotations(annotation_path)
    if not records:
        raise SystemExit("Annotation file has no images.")
    prepare_output(output_dir, overwrite=args.overwrite)
    make_dataset(records, annotation_path, output_dir, args.val_ratio, args.seed, args.reviewed_only)


if __name__ == "__main__":
    main()
