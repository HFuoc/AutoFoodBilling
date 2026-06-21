from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from backend.app.services.billing import build_bill, format_bill
from backend.app.services.classifier import FoodClassifier
from backend.app.services.detector import FoodDetector
from ml.common import configure_utf8_stdout, project_path


def main() -> None:
    configure_utf8_stdout()
    parser = argparse.ArgumentParser(description="Run detector crop + CNN classify + billing.")
    parser.add_argument("image", help="Top-down tray image path.")
    parser.add_argument("--detector-model", default="ml/models/detector/best.pt")
    parser.add_argument("--cnn-model", default="ml/models/cnn/best.onnx")
    parser.add_argument("--menu", default="configs/menu.json")
    parser.add_argument("--crop-output", default="data/generated/crops")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--json", action="store_true", help="Print JSON instead of human-readable bill.")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.is_absolute():
        image_path = project_path(image_path)
    if not image_path.exists():
        raise SystemExit(f"Input image not found: {image_path}")

    detector = FoodDetector(project_path(args.detector_model), confidence=args.confidence)
    classifier = FoodClassifier(project_path(args.cnn_model))

    crop_dir = project_path(args.crop_output)
    crops = detector.crop_detections(image_path, crop_dir)
    predictions = []
    labels = []
    for crop in crops:
        prediction = classifier.predict_image(crop["image"], top_k=args.top_k)
        labels.append(prediction["label"])
        predictions.append(
            {
                "index": crop["index"],
                "box": crop["box"],
                "detector_confidence": crop["confidence"],
                "crop_path": crop["path"],
                "prediction": prediction,
            }
        )

    bill = build_bill(labels, project_path(args.menu))
    payload = {
        "image": str(image_path),
        "detections": predictions,
        "bill": bill,
    }

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(format_bill(bill))


if __name__ == "__main__":
    main()
