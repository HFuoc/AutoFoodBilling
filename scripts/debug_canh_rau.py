"""Debug script: analyze what the model predicts for each cell of a tray image.

Usage:
    python scripts/debug_canh_rau.py <path_to_tray_image>

Shows color features and CNN predictions for every cell so you can see
exactly why "Canh rau" is or isn't being detected.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from backend.app.main import (
    color_feature_ratios,
    crop_best_standard_template_cells,
    looks_like_murky_soup,
    looks_like_vegetable_soup,
    looks_like_watery_soup,
)
from backend.app.services.classifier import FoodClassifier
from ml.common import configure_utf8_stdout, project_path


def main() -> None:
    configure_utf8_stdout()
    if len(sys.argv) < 2:
        print("Usage: python scripts/debug_canh_rau.py <image_path>")
        sys.exit(1)

    image_path = Path(sys.argv[1])
    if not image_path.is_absolute():
        image_path = project_path(image_path)
    if not image_path.exists():
        print(f"Image not found: {image_path}")
        sys.exit(1)

    cnn_model = project_path("ml/models/cnn/cell_best.onnx")
    classifier = FoodClassifier(cnn_model)
    crop_dir = project_path("data/generated/debug_crops") / image_path.stem

    print(f"\n{'='*60}")
    print(f"  DEBUG: {image_path.name}")
    print(f"{'='*60}\n")

    crops = crop_best_standard_template_cells(image_path, crop_dir, classifier)

    for crop in crops:
        cell_id = crop.get("cell_id", "?")
        prediction = classifier.predict_image(crop["image"], top_k=5)
        features = color_feature_ratios(crop["image"])

        print(f"--- Cell: {cell_id} ---")
        print(f"  CNN label:      {prediction['label']}")
        print(f"  CNN confidence:  {prediction['confidence']:.4f}")
        print(f"  Top-5:")
        for tk in prediction.get("top_k", []):
            print(f"    {tk['label']:25s} {tk['confidence']:.4f}")

        print(f"\n  Color features:")
        for name, value in sorted(features.items()):
            bar = "█" * int(value * 50)
            print(f"    {name:12s}: {value:.4f}  {bar}")

        is_veg = looks_like_vegetable_soup(features)
        is_murky = looks_like_murky_soup(features)
        is_watery = looks_like_watery_soup(features)
        print(f"\n  looks_like_vegetable_soup: {is_veg}")
        print(f"  looks_like_murky_soup:     {is_murky}")
        print(f"  looks_like_watery_soup:    {is_watery}")
        print()


if __name__ == "__main__":
    main()
