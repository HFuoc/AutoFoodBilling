from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.append(str(Path(__file__).resolve().parents[2]))

from ml.common import configure_utf8_stdout, load_json, project_path
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


def known_menu_labels(menu_path: Path, include_empty: bool) -> set[str]:
    menu = load_json(menu_path)
    labels = set(menu.get("items", {}).keys())
    if include_empty:
        labels.add("empty")
    return labels


def validate_annotations(
    records: list[dict[str, Any]],
    annotation_path: Path,
    menu_labels: set[str],
    allow_unknown_labels: bool,
) -> tuple[Counter[str], list[str]]:
    Image, ImageOps = require_pillow()
    counts: Counter[str] = Counter()
    issues: list[str] = []
    seen_sources: set[Path] = set()

    for image_index, record in enumerate(records, start=1):
        raw_file = record.get("file") or record.get("image") or record.get("path")
        cells = record.get("cells")
        if not raw_file:
            issues.append(f"image[{image_index}] missing file/image/path")
            continue
        if not isinstance(cells, list) or not cells:
            issues.append(f"{raw_file}: missing or empty cells list")
            continue

        image_path = resolve_image_path(str(raw_file), annotation_path)
        if not image_path.exists():
            issues.append(f"{raw_file}: image not found at {image_path}")
            continue
        seen_sources.add(image_path)

        try:
            with Image.open(image_path) as image:
                image = ImageOps.exif_transpose(image).convert("RGB")
                width, height = image.size
        except Exception as exc:  # noqa: BLE001 - keep data validation resilient.
            issues.append(f"{raw_file}: failed to open image: {exc}")
            continue

        for cell_index, cell in enumerate(cells, start=1):
            label = cell.get("label")
            if not label:
                issues.append(f"{raw_file} cell[{cell_index}]: missing label")
            else:
                counts[str(label)] += 1
                if not allow_unknown_labels and label not in menu_labels:
                    issues.append(f"{raw_file} cell[{cell_index}]: unknown label '{label}'")

            if "box" not in cell:
                issues.append(f"{raw_file} cell[{cell_index}]: missing box")
                continue
            try:
                box = clamp_box(list(cell["box"]), width, height)
            except Exception as exc:  # noqa: BLE001
                issues.append(f"{raw_file} cell[{cell_index}]: invalid box {cell.get('box')}: {exc}")
                continue

            x1, y1, x2, y2 = box
            box_area = (x2 - x1) * (y2 - y1)
            image_area = width * height
            if image_area and box_area / image_area < 0.01:
                issues.append(f"{raw_file} cell[{cell_index}]: box is very small: {list(box)}")

    if not seen_sources:
        issues.append("no readable annotated images")

    return counts, issues


def validate_crop_dataset(crop_root: Path, annotation_counts: Counter[str]) -> list[str]:
    issues: list[str] = []
    if not crop_root.exists():
        issues.append(f"crop dataset not found: {crop_root}")
        return issues

    classes_path = crop_root / "classes.json"
    manifest_path = crop_root / "manifest.json"
    if not classes_path.exists():
        issues.append(f"missing crop classes file: {classes_path}")
    if not manifest_path.exists():
        issues.append(f"missing crop manifest file: {manifest_path}")

    class_names: list[str] = []
    if classes_path.exists():
        try:
            class_names = list(load_json(classes_path)["classes"])
        except Exception as exc:  # noqa: BLE001
            issues.append(f"failed to read {classes_path}: {exc}")

    for label in annotation_counts:
        if class_names and label not in class_names:
            issues.append(f"crop classes.json missing annotation label: {label}")

    for class_name in class_names:
        class_dir = crop_root / class_name
        if not class_dir.exists():
            issues.append(f"class listed but folder missing: {class_name}")
            continue
        image_count = sum(1 for path in class_dir.rglob("*") if path.is_file())
        if image_count == 0:
            issues.append(f"class folder has no crop images: {class_name}")

    return issues


def print_counts(counts: Counter[str], min_per_class: int) -> list[str]:
    issues: list[str] = []
    print("Label counts")
    print("------------")
    if not counts:
        print("(none)")
        issues.append("no labeled cells")
        return issues

    for label, count in sorted(counts.items()):
        print(f"{label}: {count}")
        if count < min_per_class:
            issues.append(f"label '{label}' has only {count} cells; target at least {min_per_class}")
    return issues


def main() -> None:
    configure_utf8_stdout()
    parser = argparse.ArgumentParser(description="Check tray-cell annotations and crop dataset readiness.")
    parser.add_argument("--annotations", default="data/annotations/tray_cells.json")
    parser.add_argument("--menu", default="configs/menu.json")
    parser.add_argument("--crop-root", default="data/processed/tray_cell_food_classes")
    parser.add_argument("--min-per-class", type=int, default=20)
    parser.add_argument("--include-empty", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-unknown-labels", action="store_true")
    parser.add_argument("--skip-crops", action="store_true")
    args = parser.parse_args()

    annotation_path = project_path(args.annotations)
    menu_path = project_path(args.menu)
    crop_root = project_path(args.crop_root)
    if args.min_per_class < 1:
        raise SystemExit("--min-per-class must be >= 1")
    if not annotation_path.exists():
        raise SystemExit(f"Annotation file not found: {annotation_path}")
    if not menu_path.exists():
        raise SystemExit(f"Menu file not found: {menu_path}")

    records = load_annotations(annotation_path)
    menu_labels = known_menu_labels(menu_path, include_empty=args.include_empty)
    counts, issues = validate_annotations(
        records,
        annotation_path,
        menu_labels,
        allow_unknown_labels=args.allow_unknown_labels,
    )
    issues.extend(print_counts(counts, args.min_per_class))
    if not args.skip_crops:
        issues.extend(validate_crop_dataset(crop_root, counts))

    print()
    if issues:
        print("Issues")
        print("------")
        for issue in issues:
            print(f"- {issue}")
        raise SystemExit(1)

    print("OK: tray-cell data looks ready for training.")


if __name__ == "__main__":
    main()
