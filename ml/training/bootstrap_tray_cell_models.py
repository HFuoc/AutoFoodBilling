from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from ml.common import configure_utf8_stdout, project_path


def run_command(command: list[str], dry_run: bool) -> None:
    print(" ".join(command))
    if dry_run:
        return
    subprocess.run(command, check=True)


def main() -> None:
    configure_utf8_stdout()
    parser = argparse.ArgumentParser(
        description="Bootstrap tray-cell detector and CNN models from synthetic data."
    )
    parser.add_argument("--layouts", default="data/annotations/tray_cells.json")
    parser.add_argument("--synthetic-output", default="data/generated/tray_cell_synthetic")
    parser.add_argument("--samples", type=int, default=300)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--empty-probability", type=float, default=0.12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument("--detector-output", default="ml/models/detector/cell_best.pt")
    parser.add_argument("--detector-base-model", default="detector_nano.det")
    parser.add_argument("--detector-epochs", type=int, default=30)
    parser.add_argument("--detector-image-size", type=int, default=640)
    parser.add_argument("--detector-batch-size", type=int, default=8)
    parser.add_argument("--detector-workers", type=int, default=0)
    parser.add_argument("--detector-project", default="ml/models/detector/runs")
    parser.add_argument("--detector-run-name", default="tray_cell_synthetic")

    parser.add_argument("--cnn-output", default="ml/models/cnn/cell_best.onnx")
    parser.add_argument("--cnn-epochs", type=int, default=20)
    parser.add_argument("--cnn-batch-size", type=int, default=16)
    parser.add_argument("--cnn-image-size", type=int, default=224)
    parser.add_argument("--cnn-lr", type=float, default=3e-4)
    parser.add_argument("--cnn-val-ratio", type=float, default=0.2)

    parser.add_argument("--skip-detector", action="store_true")
    parser.add_argument("--skip-cnn", action="store_true")
    parser.add_argument("--skip-evaluate", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.samples < 1:
        raise SystemExit("--samples must be >= 1")
    if not 0 <= args.val_ratio < 1:
        raise SystemExit("--val-ratio must be >= 0 and < 1")
    if not 0 <= args.empty_probability <= 1:
        raise SystemExit("--empty-probability must be between 0 and 1")
    if args.detector_epochs < 1:
        raise SystemExit("--detector-epochs must be >= 1")
    if args.cnn_epochs < 1:
        raise SystemExit("--cnn-epochs must be >= 1")

    synthetic_output = project_path(args.synthetic_output)
    detector_data = synthetic_output / "detector" / "data.yaml"
    cnn_data = synthetic_output / "cnn"
    cnn_classes = cnn_data / "classes.json"
    val_annotations = synthetic_output / "metadata" / "synthetic_cells_val.json"

    generate_command = [
        sys.executable,
        "ml/training/generate_synthetic_tray_cell_dataset.py",
        "--layouts",
        args.layouts,
        "--output",
        args.synthetic_output,
        "--samples",
        str(args.samples),
        "--val-ratio",
        str(args.val_ratio),
        "--empty-probability",
        str(args.empty_probability),
        "--seed",
        str(args.seed),
    ]
    if args.overwrite:
        generate_command.append("--overwrite")
    run_command(generate_command, args.dry_run)

    if not args.skip_detector:
        run_command(
            [
                sys.executable,
                "ml/training/train_detector.py",
                "--data",
                str(detector_data),
                "--base-model",
                args.detector_base_model,
                "--epochs",
                str(args.detector_epochs),
                "--image-size",
                str(args.detector_image_size),
                "--batch-size",
                str(args.detector_batch_size),
                "--workers",
                str(args.detector_workers),
                "--project",
                args.detector_project,
                "--name",
                args.detector_run_name,
                "--output",
                args.detector_output,
            ],
            args.dry_run,
        )

    if not args.skip_cnn:
        run_command(
            [
                sys.executable,
                "ml/training/train_cnn.py",
                "--data-root",
                str(cnn_data),
                "--classes",
                str(cnn_classes),
                "--output",
                args.cnn_output,
                "--epochs",
                str(args.cnn_epochs),
                "--batch-size",
                str(args.cnn_batch_size),
                "--image-size",
                str(args.cnn_image_size),
                "--lr",
                str(args.cnn_lr),
                "--val-ratio",
                str(args.cnn_val_ratio),
                "--seed",
                str(args.seed),
            ],
            args.dry_run,
        )

    if not args.skip_evaluate:
        eval_command = [
            sys.executable,
            "ml/inference/evaluate_cell_pipeline.py",
            "--annotations",
            str(val_annotations),
            "--cnn-model",
            args.cnn_output,
        ]
        if not args.skip_detector:
            eval_command.extend(["--cell-detector-model", args.detector_output, "--crop-mode", "detector"])
        else:
            eval_command.extend(["--crop-mode", "template"])
        run_command(eval_command, args.dry_run)


if __name__ == "__main__":
    main()
