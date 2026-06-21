from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.append(str(Path(__file__).resolve().parents[2]))

from backend.app.services.detector import FoodDetector
from ml.common import configure_utf8_stdout, iter_images, project_path
from ml.training.preview_tray_cell_annotations import make_previews


CELL_IDS = [
    "cell_01",
    "cell_02",
    "cell_03",
    "cell_04",
    "cell_05",
    "cell_06",
    "cell_07",
    "cell_08",
    "cell_09",
    "cell_10",
    "cell_11",
    "cell_12",
]


def load_existing_annotations(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    records = payload.get("images", payload) if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        raise SystemExit(f"Invalid annotation file: {path}")

    by_source: dict[str, dict[str, Any]] = {}
    for record in records:
        raw_file = record.get("file") or record.get("image") or record.get("path")
        if raw_file:
            by_source[normalize_source_key(str(raw_file))] = record
    return by_source


def normalize_source_key(raw_path: str) -> str:
    path = Path(raw_path)
    resolved = path if path.is_absolute() else project_path(path)
    return str(resolved.resolve()).lower()


def project_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(project_path().resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def make_cell_records(detections: list[Any], label: str) -> list[dict[str, Any]]:
    cells = []
    for index, detection in enumerate(detections, start=1):
        cell_id = CELL_IDS[index - 1] if index <= len(CELL_IDS) else f"cell_{index:02d}"
        cells.append(
            {
                "cell_id": cell_id,
                "box": list(detection.box),
                "label": label,
                "confidence": round(float(detection.confidence), 4),
            }
        )
    return cells


def collect_records(
    empty_dir: Path,
    food_dir: Path,
    existing_annotations: Path,
    detector_model: Path,
    confidence: float,
    accept_suggestions: bool,
    max_detections: int,
) -> list[dict[str, Any]]:
    existing = load_existing_annotations(existing_annotations)
    detector = FoodDetector(detector_model, confidence=confidence)

    records: list[dict[str, Any]] = []
    for root, label in [(empty_dir, "empty"), (food_dir, "tray_with_food")]:
        for image_path in iter_images(root):
            source = project_relative(image_path)
            existing_record = existing.get(normalize_source_key(source))
            if existing_record and isinstance(existing_record.get("cells"), list):
                record = dict(existing_record)
                record["file"] = source
                record["reviewed"] = bool(existing_record.get("reviewed", True))
                record["annotation_source"] = "existing"
                records.append(record)
                continue

            detections = detector.detect(image_path)
            if max_detections > 0:
                detections = detections[:max_detections]
            cells = make_cell_records(detections, label)
            records.append(
                {
                    "file": source,
                    "reviewed": accept_suggestions,
                    "annotation_source": "detector_suggestion",
                    "needs_review": not accept_suggestions,
                    "cells": cells,
                }
            )
    return records


def write_annotations(records: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps({"images": records}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def print_summary(records: list[dict[str, Any]]) -> None:
    reviewed = sum(1 for record in records if record.get("reviewed") is True)
    suggested = sum(1 for record in records if record.get("annotation_source") == "detector_suggestion")
    no_cells = [record["file"] for record in records if not record.get("cells")]
    cell_counts: dict[int, int] = {}
    for record in records:
        count = len(record.get("cells", []))
        cell_counts[count] = cell_counts.get(count, 0) + 1

    print(f"images: {len(records)}")
    print(f"reviewed: {reviewed}")
    print(f"detector_suggestions: {suggested}")
    print("cell_counts:")
    for count in sorted(cell_counts):
        print(f"  {count}: {cell_counts[count]}")
    if no_cells:
        print("needs manual annotation, no cells detected:")
        for source in no_cells:
            print(f"  - {source}")


def main() -> None:
    configure_utf8_stdout()
    parser = argparse.ArgumentParser(
        description="Seed tray-cell annotations from empty_trays and tray_with_food images."
    )
    parser.add_argument("--empty-dir", default="data/raw/empty_trays")
    parser.add_argument("--food-dir", default="data/raw/tray_with_food")
    parser.add_argument("--existing-annotations", default="data/annotations/tray_cells.json")
    parser.add_argument("--output", default="data/annotations/tray_cells_auto.json")
    parser.add_argument("--preview-output", default="data/generated/annotation_previews_auto")
    parser.add_argument("--detector-model", default="ml/models/detector/cell_best.pt")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--max-detections", type=int, default=12)
    parser.add_argument(
        "--accept-suggestions",
        action="store_true",
        help="Mark detector suggestions as reviewed=true. Use only after preview quality is acceptable.",
    )
    args = parser.parse_args()

    empty_dir = project_path(args.empty_dir)
    food_dir = project_path(args.food_dir)
    output_path = project_path(args.output)
    preview_output = project_path(args.preview_output)
    detector_model = project_path(args.detector_model)
    existing_annotations = project_path(args.existing_annotations)

    if not empty_dir.exists():
        raise SystemExit(f"Missing empty tray directory: {empty_dir}")
    if not food_dir.exists():
        raise SystemExit(f"Missing tray-with-food directory: {food_dir}")
    if not detector_model.exists():
        raise SystemExit(f"Missing seed detector model: {detector_model}")
    if not 0 <= args.confidence <= 1:
        raise SystemExit("--confidence must be between 0 and 1")
    if args.max_detections < 0:
        raise SystemExit("--max-detections must be >= 0")

    records = collect_records(
        empty_dir=empty_dir,
        food_dir=food_dir,
        existing_annotations=existing_annotations,
        detector_model=detector_model,
        confidence=args.confidence,
        accept_suggestions=args.accept_suggestions,
        max_detections=args.max_detections,
    )
    write_annotations(records, output_path)
    make_previews(records, output_path, preview_output, max_images=None)
    print_summary(records)
    print(f"annotations: {output_path}")
    print(f"previews: {preview_output}")


if __name__ == "__main__":
    main()
