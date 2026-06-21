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
        description="Seed annotations, prepare detector data, and train the tray-cell detector."
    )
    parser.add_argument("--annotations", default="data/annotations/tray_cells_auto.json")
    parser.add_argument("--base-annotations", default="data/annotations/tray_cells.json")
    parser.add_argument("--empty-dir", default="data/raw/empty_trays")
    parser.add_argument("--food-dir", default="data/raw/tray_with_food")
    parser.add_argument("--seed-model", default="ml/models/detector/cell_best.pt")
    parser.add_argument("--seed-confidence", type=float, default=0.25)
    parser.add_argument("--skip-seed", action="store_true")
    parser.add_argument("--accept-suggestions", action="store_true")
    parser.add_argument("--allow-unreviewed", action="store_true")

    parser.add_argument("--dataset-output", default="data/processed/tray_cells_detector_auto")
    parser.add_argument("--preview-output", default="data/generated/annotation_previews_auto")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--base-model", default="detector_nano.det")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--image-size", type=int, default=768)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--project", default="ml/models/detector/runs")
    parser.add_argument("--name", default="tray_cell_retrain")
    parser.add_argument("--output", default="ml/models/detector/cell_best.pt")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--iou-threshold", type=float, default=0.5)

    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-evaluate", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.epochs < 1:
        raise SystemExit("--epochs must be >= 1")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be >= 1")
    if not 0 <= args.val_ratio < 1:
        raise SystemExit("--val-ratio must be >= 0 and < 1")

    annotation_path = project_path(args.annotations)
    dataset_output = project_path(args.dataset_output)

    if not args.skip_seed:
        seed_command = [
            sys.executable,
            "ml/training/seed_tray_cell_annotations.py",
            "--empty-dir",
            args.empty_dir,
            "--food-dir",
            args.food_dir,
            "--existing-annotations",
            args.base_annotations,
            "--output",
            args.annotations,
            "--preview-output",
            args.preview_output,
            "--detector-model",
            args.seed_model,
            "--confidence",
            str(args.seed_confidence),
        ]
        if args.accept_suggestions:
            seed_command.append("--accept-suggestions")
        run_command(seed_command, args.dry_run)

    if not args.dry_run and not annotation_path.exists():
        raise SystemExit(f"Annotation file not found: {annotation_path}")

    prepare_command = [
        sys.executable,
        "ml/training/prepare_tray_cell_detector_dataset.py",
        "--annotations",
        args.annotations,
        "--output",
        args.dataset_output,
        "--val-ratio",
        str(args.val_ratio),
        "--seed",
        str(args.seed),
    ]
    if args.overwrite:
        prepare_command.append("--overwrite")
    if not args.allow_unreviewed:
        prepare_command.append("--reviewed-only")
    run_command(prepare_command, args.dry_run)

    data_yaml = dataset_output / "data.yaml"
    if not args.skip_train:
        run_command(
            [
                sys.executable,
                "ml/training/train_detector.py",
                "--data",
                str(data_yaml),
                "--base-model",
                args.base_model,
                "--epochs",
                str(args.epochs),
                "--image-size",
                str(args.image_size),
                "--batch-size",
                str(args.batch_size),
                "--workers",
                str(args.workers),
                "--project",
                args.project,
                "--name",
                args.name,
                "--output",
                args.output,
            ],
            args.dry_run,
        )

    if not args.skip_evaluate:
        run_command(
            [
                sys.executable,
                "ml/inference/evaluate_cell_pipeline.py",
                "--annotations",
                args.annotations,
                "--cell-detector-model",
                args.output,
                "--confidence",
                str(args.confidence),
                "--iou-threshold",
                str(args.iou_threshold),
                "--skip-classifier",
            ],
            args.dry_run,
        )


if __name__ == "__main__":
    main()
