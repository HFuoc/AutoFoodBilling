from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))

from ml.common import project_path


def require_detector_runtime():
    detector_config_dir = project_path("data/generated/detector_runtime")
    mpl_config_dir = project_path("data/generated/matplotlib")
    detector_config_dir.mkdir(parents=True, exist_ok=True)
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")
    os.environ.setdefault("detector_CONFIG_DIR", str(detector_config_dir))
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))
    try:
        from detector_runtime import detector
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: detector_runtime. Install with: "
            "pip install -r backend/requirements.txt"
        ) from exc
    return detector


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one-class detector food_item detector.")
    parser.add_argument("--data", default="data/generated/detector_synthetic/data.yaml")
    parser.add_argument("--base-model", default="detector_nano.det")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--project", default="ml/models/detector/runs")
    parser.add_argument("--name", default="food_item")
    parser.add_argument("--output", default="ml/models/detector/best.pt")
    args = parser.parse_args()

    detector = require_detector_runtime()
    data_path = project_path(args.data)
    if not data_path.exists():
        raise SystemExit(
            f"detector data config not found: {data_path}. "
            "Run ml/training/generate_detector_dataset.py first."
        )

    model = detector(args.base_model)
    result = model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.image_size,
        batch=args.batch_size,
        workers=args.workers,
        project=str(project_path(args.project)),
        name=args.name,
    )

    best_path = Path(result.save_dir) / "weights" / "best.pt"
    output_path = project_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_path, output_path)
    print(f"Saved detector best model: {output_path}")


if __name__ == "__main__":
    main()
