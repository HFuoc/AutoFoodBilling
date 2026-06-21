from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.append(str(Path(__file__).resolve().parents[2]))

from ml.common import configure_utf8_stdout, iter_images, load_classes, project_path
from ml.training.extract_tray_cell_crops import safe_name
from ml.training.prepare_tray_cell_detector_dataset import (
    clamp_box,
    load_annotations,
    resolve_image_path,
    to_detector_line,
)


def require_pillow():
    try:
        from PIL import Image, ImageEnhance, ImageFilter, ImageOps
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: pillow. Install dependencies with: "
            "pip install -r backend/requirements.txt"
        ) from exc
    return Image, ImageEnhance, ImageFilter, ImageOps


def collect_food_images_by_class(food_root: Path, class_names: list[str]) -> dict[str, list[Path]]:
    images_by_class: dict[str, list[Path]] = {}
    for class_name in class_names:
        images = iter_images(food_root / class_name)
        if images:
            images_by_class[class_name] = images
    return images_by_class


def make_soft_mask(size: tuple[int, int], feather: int):
    Image, _, ImageFilter, _ = require_pillow()
    width, height = size
    mask = Image.new("L", size, 0)
    inner = Image.new("L", (max(1, width - feather * 2), max(1, height - feather * 2)), 255)
    mask.paste(inner, (feather, feather))
    return mask.filter(ImageFilter.GaussianBlur(radius=max(1, feather // 2)))


def transform_food_for_cell(food_path: Path, cell_size: tuple[int, int]):
    Image, ImageEnhance, _, ImageOps = require_pillow()
    cell_w, cell_h = cell_size
    with Image.open(food_path) as food:
        food = ImageOps.exif_transpose(food).convert("RGB")

    target_w = max(12, int(cell_w * random.uniform(0.74, 1.02)))
    target_h = max(12, int(cell_h * random.uniform(0.74, 1.02)))
    scale = min(target_w / food.width, target_h / food.height)
    new_size = (max(12, int(food.width * scale)), max(12, int(food.height * scale)))
    food = food.resize(new_size)

    angle = random.uniform(-8, 8)
    food = food.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC)

    brightness = random.uniform(0.86, 1.16)
    contrast = random.uniform(0.90, 1.12)
    saturation = random.uniform(0.90, 1.12)
    food = ImageEnhance.Brightness(food).enhance(brightness)
    food = ImageEnhance.Contrast(food).enhance(contrast)
    food = ImageEnhance.Color(food).enhance(saturation)

    if food.width > cell_w or food.height > cell_h:
        scale = min(cell_w / food.width, cell_h / food.height)
        food = food.resize((max(12, int(food.width * scale)), max(12, int(food.height * scale))))

    feather = max(3, min(food.size) // 16)
    mask = make_soft_mask(food.size, feather)
    return food, mask


def paste_food_in_cell(
    canvas: Any,
    box: tuple[int, int, int, int],
    food_path: Path,
) -> None:
    x1, y1, x2, y2 = box
    cell_w, cell_h = x2 - x1, y2 - y1
    food, mask = transform_food_for_cell(food_path, (cell_w, cell_h))
    max_dx = max(0, cell_w - food.width)
    max_dy = max(0, cell_h - food.height)
    offset_x = x1 + (random.randint(0, max_dx) if max_dx else 0)
    offset_y = y1 + (random.randint(0, max_dy) if max_dy else 0)
    canvas.paste(food, (offset_x, offset_y), mask)


def prepare_output(output_dir: Path, overwrite: bool) -> tuple[Path, Path]:
    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)
    detector_dir = output_dir / "detector"
    cnn_dir = output_dir / "cnn"
    for split in ["train", "val"]:
        (detector_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (detector_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
    (output_dir / "metadata").mkdir(parents=True, exist_ok=True)
    cnn_dir.mkdir(parents=True, exist_ok=True)
    return detector_dir, cnn_dir


def write_detector_data_yaml(detector_dir: Path) -> None:
    content = "\n".join(
        [
            f"path: {detector_dir.as_posix()}",
            "train: images/train",
            "val: images/val",
            "names:",
            "  0: tray_cell",
            "",
        ]
    )
    (detector_dir / "data.yaml").write_text(content, encoding="utf-8")


def build_layouts(
    records: list[dict[str, Any]],
    layouts_path: Path,
    strict_missing: bool,
) -> list[dict[str, Any]]:
    layouts: list[dict[str, Any]] = []
    for record in records:
        raw_file = record.get("file") or record.get("image") or record.get("path")
        cells = record.get("cells")
        if not raw_file or not isinstance(cells, list) or not cells:
            raise SystemExit("Each layout record needs 'file' and a non-empty 'cells' list.")
        image_path = resolve_image_path(str(raw_file), layouts_path)
        if not image_path.exists():
            message = f"Layout image not found, skipping: {image_path}"
            if strict_missing:
                raise SystemExit(message)
            print(f"WARNING: {message}")
            continue
        layouts.append({"image_path": image_path, "cells": cells})
    if not layouts:
        raise SystemExit("No usable tray layouts found.")
    return layouts


def generate_dataset(
    layouts: list[dict[str, Any]],
    food_images: dict[str, list[Path]],
    output_dir: Path,
    samples: int,
    val_ratio: float,
    empty_probability: float,
    include_empty_crops: bool,
    overwrite: bool,
) -> None:
    Image, _, _, ImageOps = require_pillow()
    detector_dir, cnn_dir = prepare_output(output_dir, overwrite)
    class_names = list(food_images.keys())
    crop_counts: Counter[str] = Counter()
    metadata: list[dict[str, Any]] = []
    crop_manifest: list[dict[str, Any]] = []
    val_count = int(round(samples * val_ratio))

    for index in range(samples):
        split = "val" if index < val_count else "train"
        layout = random.choice(layouts)
        stem = f"synthetic_cell_{index:05d}"
        image_path = detector_dir / "images" / split / f"{stem}.jpg"
        label_path = detector_dir / "labels" / split / f"{stem}.txt"
        with Image.open(layout["image_path"]) as tray:
            canvas = ImageOps.exif_transpose(tray).convert("RGB")
        width, height = canvas.size

        detector_lines: list[str] = []
        cell_records: list[dict[str, Any]] = []

        for cell_index, cell in enumerate(layout["cells"], start=1):
            box = clamp_box(list(cell["box"]), width, height)
            detector_lines.append(to_detector_line(box, (width, height)))

            is_empty = random.random() < empty_probability
            label = "empty" if is_empty else random.choice(class_names)
            if not is_empty:
                paste_food_in_cell(canvas, box, random.choice(food_images[label]))

            if include_empty_crops or not is_empty:
                class_dir = cnn_dir / label
                class_dir.mkdir(parents=True, exist_ok=True)
                crop = canvas.crop(box)
                crop_counts[label] += 1
                crop_name = f"synthetic_{index:05d}_{safe_name(str(cell.get('cell_id', f'cell_{cell_index:02d}')))}.jpg"
                crop_path = class_dir / crop_name
                crop.save(crop_path, quality=95)
                crop_manifest.append(
                    {
                        "source": str(image_path),
                        "crop": str(crop_path),
                        "label": label,
                        "cell_id": cell.get("cell_id", f"cell_{cell_index:02d}"),
                        "box": list(box),
                        "split": split,
                    }
                )
            else:
                crop_path = None

            cell_records.append(
                {
                    "cell_index": cell_index,
                    "cell_id": cell.get("cell_id", f"cell_{cell_index:02d}"),
                    "box": list(box),
                    "label": label,
                    "crop": str(crop_path) if crop_path else None,
                }
            )

        canvas.save(image_path, quality=95)
        label_path.write_text("\n".join(detector_lines) + "\n", encoding="utf-8")
        metadata.append(
            {
                "source_layout": str(layout["image_path"]),
                "split": split,
                "image": str(image_path),
                "label_file": str(label_path),
                "cells": cell_records,
            }
        )

    write_detector_data_yaml(detector_dir)
    (cnn_dir / "classes.json").write_text(
        json.dumps({"classes": list(crop_counts.keys())}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (cnn_dir / "manifest.json").write_text(
        json.dumps({"crops": crop_manifest}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "metadata" / "synthetic_cells.json").write_text(
        json.dumps({"images": metadata}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    for split in ["train", "val"]:
        split_metadata = [record for record in metadata if record["split"] == split]
        (output_dir / "metadata" / f"synthetic_cells_{split}.json").write_text(
            json.dumps({"images": split_metadata}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print(f"Generated {samples} synthetic tray-cell images at {output_dir}")
    print(f"detector config: {detector_dir / 'data.yaml'}")
    print(f"CNN data root: {cnn_dir}")
    print(f"CNN classes: {cnn_dir / 'classes.json'}")
    print(f"Synthetic val annotations: {output_dir / 'metadata' / 'synthetic_cells_val.json'}")
    for label, count in sorted(crop_counts.items()):
        print(f"{label}: {count}")


def main() -> None:
    configure_utf8_stdout()
    parser = argparse.ArgumentParser(description="Generate synthetic tray-cell detector and CNN datasets.")
    parser.add_argument("--layouts", default="data/annotations/tray_cells.json")
    parser.add_argument("--food-root", default="data/raw/food_classes")
    parser.add_argument("--classes", default="configs/classes.json")
    parser.add_argument("--output", default="data/generated/tray_cell_synthetic")
    parser.add_argument("--samples", type=int, default=300)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--empty-probability", type=float, default=0.12)
    parser.add_argument("--skip-empty-crops", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--strict-layouts", action="store_true")
    args = parser.parse_args()

    if args.samples < 1:
        raise SystemExit("--samples must be >= 1")
    if not 0 <= args.val_ratio < 1:
        raise SystemExit("--val-ratio must be >= 0 and < 1")
    if not 0 <= args.empty_probability <= 1:
        raise SystemExit("--empty-probability must be between 0 and 1")

    random.seed(args.seed)
    layouts_path = project_path(args.layouts)
    food_root = project_path(args.food_root)
    output_dir = project_path(args.output)
    if not layouts_path.exists():
        raise SystemExit(f"Layout annotation file not found: {layouts_path}")

    class_names = load_classes(args.classes)
    food_images = collect_food_images_by_class(food_root, class_names)
    if not food_images:
        raise SystemExit(f"No food images found under {food_root}")

    layouts = build_layouts(load_annotations(layouts_path), layouts_path, args.strict_layouts)
    generate_dataset(
        layouts=layouts,
        food_images=food_images,
        output_dir=output_dir,
        samples=args.samples,
        val_ratio=args.val_ratio,
        empty_probability=args.empty_probability,
        include_empty_crops=not args.skip_empty_crops,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
