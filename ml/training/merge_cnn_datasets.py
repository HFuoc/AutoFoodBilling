from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from ml.common import configure_utf8_stdout, iter_images, load_classes, project_path
from ml.training.extract_tray_cell_crops import safe_name


def prepare_output(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def discover_classes(input_dirs: list[Path], class_config: str | None) -> list[str]:
    if class_config:
        return load_classes(class_config)

    class_names: list[str] = []
    seen: set[str] = set()
    for input_dir in input_dirs:
        classes_path = input_dir / "classes.json"
        if classes_path.exists():
            payload = json.loads(classes_path.read_text(encoding="utf-8"))
            candidates = payload.get("classes", [])
        else:
            candidates = [path.name for path in input_dir.iterdir() if path.is_dir()]
        for class_name in candidates:
            if class_name not in seen:
                seen.add(class_name)
                class_names.append(class_name)
    return class_names


def main() -> None:
    configure_utf8_stdout()
    parser = argparse.ArgumentParser(description="Merge folder-per-class CNN image datasets.")
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", default="data/processed/tray_cell_food_classes_combined")
    parser.add_argument("--classes", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    input_dirs = [project_path(path) for path in args.inputs]
    for input_dir in input_dirs:
        if not input_dir.exists():
            raise SystemExit(f"Input CNN dataset not found: {input_dir}")

    output_dir = project_path(args.output)
    prepare_output(output_dir, args.overwrite)
    class_names = discover_classes(input_dirs, args.classes)
    if not class_names:
        raise SystemExit("No classes found.")

    counts = {class_name: 0 for class_name in class_names}
    manifest = []
    for input_index, input_dir in enumerate(input_dirs, start=1):
        for class_name in class_names:
            class_dir = input_dir / class_name
            if not class_dir.exists():
                continue
            output_class_dir = output_dir / class_name
            output_class_dir.mkdir(parents=True, exist_ok=True)
            for image_path in iter_images(class_dir):
                counts[class_name] += 1
                new_name = f"d{input_index:02d}_{safe_name(image_path.stem)}_{counts[class_name]:06d}{image_path.suffix.lower()}"
                output_path = output_class_dir / new_name
                shutil.copy2(image_path, output_path)
                manifest.append(
                    {
                        "source": str(image_path),
                        "path": str(output_path),
                        "label": class_name,
                    }
                )

    copied = sum(counts.values())
    if copied == 0:
        raise SystemExit("No images were copied.")

    (output_dir / "classes.json").write_text(
        json.dumps({"classes": class_names}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "manifest.json").write_text(
        json.dumps({"images": manifest}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Merged {copied} CNN images into {output_dir}")
    for class_name in class_names:
        print(f"{class_name}: {counts[class_name]}")
    print(f"Class config: {output_dir / 'classes.json'}")


if __name__ == "__main__":
    main()
