from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from ml.common import configure_utf8_stdout, project_path


def prepare_output(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)
    for split in ["train", "val"]:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)


def write_data_yaml(output_dir: Path) -> None:
    content = "\n".join(
        [
            f"path: {output_dir.as_posix()}",
            "train: images/train",
            "val: images/val",
            "names:",
            "  0: tray_cell",
            "",
        ]
    )
    (output_dir / "data.yaml").write_text(content, encoding="utf-8")


def copy_split(input_dir: Path, output_dir: Path, split: str, prefix: str) -> int:
    image_dir = input_dir / "images" / split
    label_dir = input_dir / "labels" / split
    if not image_dir.exists() or not label_dir.exists():
        return 0

    count = 0
    for image_path in sorted(path for path in image_dir.iterdir() if path.is_file()):
        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue
        stem = f"{prefix}_{split}_{count:06d}"
        new_image = output_dir / "images" / split / f"{stem}{image_path.suffix.lower()}"
        new_label = output_dir / "labels" / split / f"{stem}.txt"
        shutil.copy2(image_path, new_image)
        shutil.copy2(label_path, new_label)
        count += 1
    return count


def main() -> None:
    configure_utf8_stdout()
    parser = argparse.ArgumentParser(description="Merge one-class tray-cell detector datasets.")
    parser.add_argument("--inputs", nargs="+", required=True, help="detector dataset roots containing data.yaml.")
    parser.add_argument("--output", default="data/processed/tray_cells_detector_combined")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_dir = project_path(args.output)
    prepare_output(output_dir, args.overwrite)

    total = 0
    for index, raw_input in enumerate(args.inputs, start=1):
        input_dir = project_path(raw_input)
        if not input_dir.exists():
            raise SystemExit(f"Input detector dataset not found: {input_dir}")
        prefix = f"d{index:02d}"
        for split in ["train", "val"]:
            count = copy_split(input_dir, output_dir, split, prefix)
            print(f"{input_dir} {split}: {count}")
            total += count

    if total == 0:
        raise SystemExit("No image/label pairs were copied.")

    write_data_yaml(output_dir)
    print(f"Merged {total} detector image/label pairs into {output_dir}")
    print(f"detector config: {output_dir / 'data.yaml'}")


if __name__ == "__main__":
    main()
