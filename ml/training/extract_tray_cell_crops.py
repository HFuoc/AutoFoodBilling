from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.append(str(Path(__file__).resolve().parents[2]))

from ml.common import configure_utf8_stdout, project_path
from ml.training.prepare_tray_cell_detector_dataset import clamp_box, load_annotations, resolve_image_path


def require_pillow():
    try:
        from PIL import Image, ImageOps
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: pillow. Install dependencies with: "
            "pip install -r backend/requirements.txt"
        ) from exc
    return Image, ImageOps


def expand_box(
    box: tuple[int, int, int, int],
    image_size: tuple[int, int],
    padding: int,
) -> tuple[int, int, int, int]:
    width, height = image_size
    x1, y1, x2, y2 = box
    return (
        max(0, x1 - padding),
        max(0, y1 - padding),
        min(width, x2 + padding),
        min(height, y2 + padding),
    )


def safe_name(value: str) -> str:
    value = value.strip()
    value = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", value)
    value = re.sub(r"\s+", "_", value)
    return value or "cell"


def prepare_output(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def extract_crops(
    records: list[dict[str, Any]],
    annotation_path: Path,
    output_dir: Path,
    padding: int,
    skip_unlabeled: bool,
) -> None:
    Image, ImageOps = require_pillow()
    counts: Counter[str] = Counter()
    manifest: list[dict[str, Any]] = []
    skipped = 0

    for image_index, record in enumerate(records, start=1):
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

            for cell_index, cell in enumerate(cells, start=1):
                label = cell.get("label")
                if not label:
                    if skip_unlabeled:
                        skipped += 1
                        continue
                    label = "unlabeled"

                box = clamp_box(list(cell["box"]), width, height)
                crop_box = expand_box(box, (width, height), padding)
                crop = image.crop(crop_box)

                label_dir = output_dir / str(label)
                label_dir.mkdir(parents=True, exist_ok=True)
                cell_id = safe_name(str(cell.get("cell_id", f"cell_{cell_index:02d}")))
                stem = f"{image_path.stem}_{image_index:04d}_{cell_id}_{counts[str(label)] + 1:04d}"
                crop_path = label_dir / f"{safe_name(stem)}.jpg"
                crop.save(crop_path, quality=95)

                counts[str(label)] += 1
                manifest.append(
                    {
                        "source": str(image_path),
                        "crop": str(crop_path),
                        "label": str(label),
                        "cell_id": cell.get("cell_id", f"cell_{cell_index:02d}"),
                        "box": list(box),
                        "crop_box": list(crop_box),
                    }
                )

    (output_dir / "manifest.json").write_text(
        json.dumps({"crops": manifest}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "classes.json").write_text(
        json.dumps({"classes": list(counts.keys())}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Wrote {sum(counts.values())} labeled cell crops to {output_dir}")
    print(f"Class config: {output_dir / 'classes.json'}")
    for label, count in sorted(counts.items()):
        print(f"{label}: {count}")
    if skipped:
        print(f"Skipped unlabeled cells: {skipped}")


def main() -> None:
    configure_utf8_stdout()
    parser = argparse.ArgumentParser(description="Extract labeled tray-cell crops for CNN training.")
    parser.add_argument("--annotations", default="data/annotations/tray_cells.json")
    parser.add_argument("--output", default="data/processed/tray_cell_food_classes")
    parser.add_argument("--padding", type=int, default=8)
    parser.add_argument("--keep-unlabeled", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    annotation_path = project_path(args.annotations)
    output_dir = project_path(args.output)
    if not annotation_path.exists():
        raise SystemExit(f"Annotation file not found: {annotation_path}")
    if args.padding < 0:
        raise SystemExit("--padding must be >= 0")

    records = load_annotations(annotation_path)
    if not records:
        raise SystemExit("Annotation file has no images.")

    prepare_output(output_dir, overwrite=args.overwrite)
    extract_crops(
        records=records,
        annotation_path=annotation_path,
        output_dir=output_dir,
        padding=args.padding,
        skip_unlabeled=not args.keep_unlabeled,
    )


if __name__ == "__main__":
    main()
