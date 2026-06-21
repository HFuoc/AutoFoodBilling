from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from backend.app.services.billing import build_bill, format_bill
from backend.app.services.classifier import FoodClassifier
from backend.app.services.detector import FoodDetector
from backend.app.services.tray_layout import crop_standard_template_cells
from ml.common import configure_utf8_stdout, project_path


def main() -> None:
    configure_utf8_stdout()
    parser = argparse.ArgumentParser(description="Run tray-cell crop + CNN food classification.")
    parser.add_argument("image", help="Top-down tray image path.")
    parser.add_argument("--cell-detector-model", default="ml/models/detector/cell_best.pt")
    parser.add_argument("--cnn-model", default="ml/models/cnn/cell_best.cnn")
    parser.add_argument("--menu", default="configs/menu.json")
    parser.add_argument("--crop-output", default="data/generated/cell_crops")
    parser.add_argument("--confidence", type=float, default=0.2)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--empty-label", default="empty")
    parser.add_argument(
        "--crop-mode",
        choices=["template", "detector"],
        default="template",
        help="template uses the fixed single-tray layout; detector uses the detector cell model.",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON instead of human-readable output.")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.is_absolute():
        image_path = project_path(image_path)
    if not image_path.exists():
        raise SystemExit(f"Input image not found: {image_path}")

    detector = None
    if args.crop_mode == "detector":
        detector = FoodDetector(project_path(args.cell_detector_model), confidence=args.confidence)
    classifier = FoodClassifier(project_path(args.cnn_model))

    crop_dir = project_path(args.crop_output) / image_path.stem
    if args.crop_mode == "template":
        crops = crop_standard_template_cells(image_path, crop_dir)
    else:
        assert detector is not None
        crops = detector.crop_detections(image_path, crop_dir)
    cells = []
    bill_labels: list[str] = []

    for crop in crops:
        prediction = classifier.predict_image(crop["image"], top_k=args.top_k)
        label = prediction["label"]
        is_empty = label == args.empty_label
        if not is_empty:
            bill_labels.append(label)
        cells.append(
            {
                "cell_index": crop["index"],
                "box": crop["box"],
                "cell_confidence": crop["confidence"],
                "crop_path": crop["path"],
                "is_empty": is_empty,
                "prediction": prediction,
            }
        )

    bill = build_bill(bill_labels, project_path(args.menu))
    payload = {
        "image": str(image_path),
        "cells": cells,
        "bill": bill,
    }

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    print("Detected tray cells:")
    for cell in cells:
        prediction = cell["prediction"]
        status = "empty" if cell["is_empty"] else prediction["label"]
        confidence = prediction["confidence"]
        print(f"{cell['cell_index']}. {status} ({confidence:.3f}) box={cell['box']}")
    print()
    print(format_bill(bill))


if __name__ == "__main__":
    main()
