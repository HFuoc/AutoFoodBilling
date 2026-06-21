from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path
from typing import Any

sys.path.append(str(Path(__file__).resolve().parents[2]))

from ml.common import configure_utf8_stdout, iter_images, load_classes, project_path
from ml.training.extract_tray_cell_crops import safe_name


def prepare_output(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def copy_images(
    image_paths: list[Path],
    output_dir: Path,
    class_name: str,
    source_name: str,
    start_index: int,
) -> list[dict[str, Any]]:
    output_class_dir = output_dir / class_name
    output_class_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []

    for offset, image_path in enumerate(image_paths, start=1):
        index = start_index + offset
        new_name = f"{source_name}_{safe_name(image_path.stem)}_{index:06d}{image_path.suffix.lower()}"
        output_path = output_class_dir / new_name
        shutil.copy2(image_path, output_path)
        manifest.append(
            {
                "source": str(image_path),
                "path": str(output_path),
                "label": class_name,
                "source_dataset": source_name,
            }
        )
    return manifest


def collect_primary_images(primary_dir: Path | None, class_name: str) -> list[Path]:
    if primary_dir is None:
        return []
    class_dir = primary_dir / class_name
    return iter_images(class_dir) if class_dir.exists() else []


def collect_supplement_images(
    supplement_dir: Path,
    class_name: str,
    max_per_class: int,
    rng: random.Random,
) -> list[Path]:
    class_dir = supplement_dir / class_name
    if not class_dir.exists():
        return []
    images = iter_images(class_dir)
    rng.shuffle(images)
    return images[:max_per_class] if max_per_class > 0 else images


def discover_primary_extra_classes(primary_dir: Path | None, class_names: list[str]) -> list[str]:
    if primary_dir is None:
        return []

    seen = set(class_names)
    extra: list[str] = []
    classes_path = primary_dir / "classes.json"
    if classes_path.exists():
        payload = json.loads(classes_path.read_text(encoding="utf-8"))
        candidates = payload.get("classes", [])
    else:
        candidates = [path.name for path in primary_dir.iterdir() if path.is_dir()]

    for class_name in candidates:
        if class_name in seen:
            continue
        class_dir = primary_dir / str(class_name)
        if class_dir.exists() and iter_images(class_dir):
            seen.add(str(class_name))
            extra.append(str(class_name))
    return extra


def main() -> None:
    configure_utf8_stdout()
    parser = argparse.ArgumentParser(
        description=(
            "Build a CNN dataset from labeled tray-cell crops plus capped "
            "single-dish food_classes images."
        )
    )
    parser.add_argument(
        "--primary",
        default="data/processed/tray_cell_food_classes_fixed",
        help="Folder-per-class tray-cell crop dataset. If missing, only supplement images are used.",
    )
    parser.add_argument("--supplement", default="data/raw/food_classes")
    parser.add_argument("--output", default="data/processed/tray_cell_food_classes_supplemented")
    parser.add_argument("--classes", default="configs/classes.json")
    parser.add_argument("--max-supplement-per-class", type=int, default=250)
    parser.add_argument(
        "--include-primary-extra-classes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Keep classes that exist only in the primary tray-cell crop dataset, "
            "such as empty. Supplement images are still copied only for configured classes."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.max_supplement_per_class < 0:
        raise SystemExit("--max-supplement-per-class must be >= 0")

    primary_dir = project_path(args.primary)
    if not primary_dir.exists():
        primary_dir = None
    supplement_dir = project_path(args.supplement)
    if not supplement_dir.exists():
        raise SystemExit(f"Supplement dataset not found: {supplement_dir}")

    output_dir = project_path(args.output)
    configured_class_names = load_classes(args.classes)
    class_names = list(configured_class_names)
    if args.include_primary_extra_classes:
        class_names.extend(discover_primary_extra_classes(primary_dir, class_names))
    if not class_names:
        raise SystemExit("No classes found in class config.")

    rng = random.Random(args.seed)
    prepare_output(output_dir, args.overwrite)

    counts: dict[str, dict[str, int]] = {
        class_name: {"primary": 0, "supplement": 0, "total": 0}
        for class_name in class_names
    }
    manifest: list[dict[str, Any]] = []

    for class_name in class_names:
        primary_images = collect_primary_images(primary_dir, class_name)
        supplement_images = collect_supplement_images(
            supplement_dir=supplement_dir,
            class_name=class_name,
            max_per_class=args.max_supplement_per_class,
            rng=rng,
        ) if class_name in configured_class_names else []

        copied_primary = copy_images(
            image_paths=primary_images,
            output_dir=output_dir,
            class_name=class_name,
            source_name="primary",
            start_index=0,
        )
        copied_supplement = copy_images(
            image_paths=supplement_images,
            output_dir=output_dir,
            class_name=class_name,
            source_name="supplement",
            start_index=len(copied_primary),
        )

        manifest.extend(copied_primary)
        manifest.extend(copied_supplement)
        counts[class_name]["primary"] = len(copied_primary)
        counts[class_name]["supplement"] = len(copied_supplement)
        counts[class_name]["total"] = len(copied_primary) + len(copied_supplement)

    copied = sum(item["total"] for item in counts.values())
    if copied == 0:
        raise SystemExit("No images were copied.")

    missing = [class_name for class_name, item in counts.items() if item["total"] == 0]
    if missing:
        print("Warning: no images for classes:")
        for class_name in missing:
            print(f"  - {class_name}")

    (output_dir / "classes.json").write_text(
        json.dumps({"classes": class_names}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "manifest.json").write_text(
        json.dumps({"images": manifest}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(
        json.dumps({"counts": counts}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Wrote {copied} CNN images to {output_dir}")
    for class_name in class_names:
        item = counts[class_name]
        print(
            f"{class_name}: total={item['total']} "
            f"primary={item['primary']} supplement={item['supplement']}"
        )
    print(f"Class config: {output_dir / 'classes.json'}")


if __name__ == "__main__":
    main()
