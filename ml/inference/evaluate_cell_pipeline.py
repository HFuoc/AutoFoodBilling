from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.append(str(Path(__file__).resolve().parents[2]))

from backend.app.services.classifier import FoodClassifier
from backend.app.services.detector import FoodDetector
from backend.app.services.tray_layout import crop_standard_template_cells
from ml.common import configure_utf8_stdout, project_path
from ml.training.prepare_tray_cell_detector_dataset import clamp_box, load_annotations, resolve_image_path


@dataclass(frozen=True)
class GroundTruthCell:
    index: int
    box: tuple[int, int, int, int]
    label: str | None
    cell_id: str


def require_pillow():
    try:
        from PIL import Image, ImageOps
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: pillow. Install dependencies with: "
            "pip install -r backend/requirements.txt"
        ) from exc
    return Image, ImageOps


def iou(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def load_gt_cells(record: dict[str, Any], annotation_path: Path, image_size: tuple[int, int]) -> list[GroundTruthCell]:
    width, height = image_size
    cells = record.get("cells")
    if not isinstance(cells, list):
        raise ValueError("record missing cells list")

    gt_cells: list[GroundTruthCell] = []
    for index, cell in enumerate(cells, start=1):
        box = clamp_box(list(cell["box"]), width, height)
        gt_cells.append(
            GroundTruthCell(
                index=index,
                box=box,
                label=cell.get("label"),
                cell_id=str(cell.get("cell_id", f"cell_{index:02d}")),
            )
        )
    return gt_cells


def match_detections(
    gt_cells: list[GroundTruthCell],
    detections: list[Any],
    iou_threshold: float,
) -> tuple[list[dict[str, Any]], list[GroundTruthCell], list[Any]]:
    candidates: list[tuple[float, int, int]] = []
    for gt_index, gt_cell in enumerate(gt_cells):
        for det_index, detection in enumerate(detections):
            candidates.append((iou(gt_cell.box, detection.box), gt_index, det_index))
    candidates.sort(reverse=True, key=lambda item: item[0])

    used_gt: set[int] = set()
    used_det: set[int] = set()
    matches: list[dict[str, Any]] = []
    for overlap, gt_index, det_index in candidates:
        if overlap < iou_threshold:
            break
        if gt_index in used_gt or det_index in used_det:
            continue
        used_gt.add(gt_index)
        used_det.add(det_index)
        matches.append(
            {
                "gt": gt_cells[gt_index],
                "detection": detections[det_index],
                "iou": overlap,
            }
        )

    missed = [cell for index, cell in enumerate(gt_cells) if index not in used_gt]
    extra = [detection for index, detection in enumerate(detections) if index not in used_det]
    return matches, missed, extra


def evaluate(
    annotations: Path,
    detector: FoodDetector,
    classifier: FoodClassifier | None,
    iou_threshold: float,
    top_k: int,
) -> dict[str, Any]:
    Image, ImageOps = require_pillow()
    records = load_annotations(annotations)

    totals = {
        "images": 0,
        "gt_cells": 0,
        "detections": 0,
        "matched_cells": 0,
        "missed_cells": 0,
        "extra_detections": 0,
        "classified_cells": 0,
        "correct_labels": 0,
    }
    image_results: list[dict[str, Any]] = []
    ious: list[float] = []

    for record in records:
        raw_file = record.get("file") or record.get("image") or record.get("path")
        if not raw_file:
            raise SystemExit("Annotation record missing file/image/path")
        image_path = resolve_image_path(str(raw_file), annotations)
        if not image_path.exists():
            raise SystemExit(f"Annotated image not found: {image_path}")

        with Image.open(image_path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            width, height = image.size
            gt_cells = load_gt_cells(record, annotations, (width, height))
            
            if detector is not None:
                detections = detector.detect(image_path)
            else:
                # Use precision template cropping
                crops = crop_standard_template_cells(image_path, Path("dummy"))
                class DummyDetection:
                    def __init__(self, box):
                        self.box = box
                        self.confidence = 1.0
                detections = [DummyDetection(c["box"]) for c in crops]
                
            matches, missed, extra = match_detections(gt_cells, detections, iou_threshold)

            matched_payload = []
            for match in matches:
                gt_cell: GroundTruthCell = match["gt"]
                detection = match["detection"]
                prediction = None
                correct = None
                if classifier is not None and gt_cell.label:
                    crop = image.crop(detection.box)
                    prediction = classifier.predict_image(crop, top_k=top_k)
                    correct = prediction["label"] == gt_cell.label
                    totals["classified_cells"] += 1
                    if correct:
                        totals["correct_labels"] += 1
                ious.append(match["iou"])
                matched_payload.append(
                    {
                        "cell_id": gt_cell.cell_id,
                        "gt_label": gt_cell.label,
                        "gt_box": gt_cell.box,
                        "detected_box": detection.box,
                        "detector_confidence": detection.confidence,
                        "iou": match["iou"],
                        "prediction": prediction,
                        "correct": correct,
                    }
                )

        totals["images"] += 1
        totals["gt_cells"] += len(gt_cells)
        totals["detections"] += len(detections)
        totals["matched_cells"] += len(matches)
        totals["missed_cells"] += len(missed)
        totals["extra_detections"] += len(extra)
        image_results.append(
            {
                "image": str(image_path),
                "matched": matched_payload,
                "missed": [{"cell_id": cell.cell_id, "label": cell.label, "box": cell.box} for cell in missed],
                "extra": [{"box": detection.box, "confidence": detection.confidence} for detection in extra],
            }
        )

    detection_recall = totals["matched_cells"] / totals["gt_cells"] if totals["gt_cells"] else 0.0
    detection_precision = totals["matched_cells"] / totals["detections"] if totals["detections"] else 0.0
    label_accuracy = (
        totals["correct_labels"] / totals["classified_cells"] if totals["classified_cells"] else None
    )
    return {
        "summary": {
            **totals,
            "iou_threshold": iou_threshold,
            "detection_recall": detection_recall,
            "detection_precision": detection_precision,
            "mean_iou": sum(ious) / len(ious) if ious else 0.0,
            "label_accuracy": label_accuracy,
        },
        "images": image_results,
    }


def main() -> None:
    configure_utf8_stdout()
    parser = argparse.ArgumentParser(description="Evaluate tray-cell detector and optional CNN classifier.")
    parser.add_argument("--annotations", default="data/annotations/tray_cells.json")
    parser.add_argument("--cell-detector-model", default="ml/models/detector/cell_best.pt")
    parser.add_argument("--cnn-model", default="ml/models/cnn/cell_best.onnx")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--crop-mode", choices=["template", "detector"], default="detector")
    parser.add_argument("--skip-classifier", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if not 0 <= args.iou_threshold <= 1:
        raise SystemExit("--iou-threshold must be between 0 and 1")

    annotations = project_path(args.annotations)
    if not annotations.exists():
        raise SystemExit(f"Annotation file not found: {annotations}")

    detector = None
    if args.crop_mode == "detector":
        detector = FoodDetector(project_path(args.cell_detector_model), confidence=args.confidence)
        
    classifier = None if args.skip_classifier else FoodClassifier(project_path(args.cnn_model))
    result = evaluate(
        annotations=annotations,
        detector=detector,
        classifier=classifier,
        iou_threshold=args.iou_threshold,
        top_k=args.top_k,
    )

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    summary = result["summary"]
    print("Tray-cell evaluation")
    print("--------------------")
    print(f"images: {summary['images']}")
    print(f"gt_cells: {summary['gt_cells']}")
    print(f"detections: {summary['detections']}")
    print(f"matched_cells: {summary['matched_cells']}")
    print(f"missed_cells: {summary['missed_cells']}")
    print(f"extra_detections: {summary['extra_detections']}")
    print(f"detection_recall: {summary['detection_recall']:.4f}")
    print(f"detection_precision: {summary['detection_precision']:.4f}")
    print(f"mean_iou: {summary['mean_iou']:.4f}")
    if summary["label_accuracy"] is not None:
        print(f"label_accuracy: {summary['label_accuracy']:.4f}")


if __name__ == "__main__":
    main()
