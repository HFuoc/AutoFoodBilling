from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from ml.common import IMAGE_EXTENSIONS, configure_utf8_stdout, iter_images, load_classes, project_path


def require_pillow():
    try:
        from PIL import Image, ImageEnhance, ImageFilter, ImageOps
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: pillow. Install dependencies with: "
            "pip install -r backend/requirements.txt"
        ) from exc
    return Image, ImageEnhance, ImageFilter, ImageOps


def collect_food_images(food_root: Path, class_names: list[str]) -> list[Path]:
    images: list[Path] = []
    for class_name in class_names:
        class_dir = food_root / class_name
        images.extend(iter_images(class_dir))
    return images


def make_soft_mask(size: tuple[int, int], feather: int):
    Image, _, ImageFilter, _ = require_pillow()
    width, height = size
    mask = Image.new("L", size, 0)
    inner = Image.new("L", (max(1, width - feather * 2), max(1, height - feather * 2)), 255)
    mask.paste(inner, (feather, feather))
    return mask.filter(ImageFilter.GaussianBlur(radius=max(1, feather // 2)))


def transform_food(food_path: Path, min_scale: float, max_scale: float, tray_size: tuple[int, int]):
    Image, ImageEnhance, _, ImageOps = require_pillow()
    tray_w, tray_h = tray_size
    with Image.open(food_path) as food:
        food = ImageOps.exif_transpose(food).convert("RGB")

    target_long = int(random.uniform(min_scale, max_scale) * min(tray_w, tray_h))
    ratio = target_long / max(food.size)
    new_size = (max(16, int(food.width * ratio)), max(16, int(food.height * ratio)))
    food = food.resize(new_size)

    angle = random.uniform(-12, 12)
    food = food.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC)

    brightness = random.uniform(0.85, 1.15)
    contrast = random.uniform(0.9, 1.1)
    food = ImageEnhance.Brightness(food).enhance(brightness)
    food = ImageEnhance.Contrast(food).enhance(contrast)

    feather = max(4, min(food.size) // 18)
    mask = make_soft_mask(food.size, feather)
    return food, mask


def overlaps(box: tuple[int, int, int, int], boxes: list[tuple[int, int, int, int]], max_iou: float) -> bool:
    x1, y1, x2, y2 = box
    area = max(0, x2 - x1) * max(0, y2 - y1)
    for ox1, oy1, ox2, oy2 in boxes:
        ix1, iy1 = max(x1, ox1), max(y1, oy1)
        ix2, iy2 = min(x2, ox2), min(y2, oy2)
        intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        other_area = max(0, ox2 - ox1) * max(0, oy2 - oy1)
        union = area + other_area - intersection
        if union and intersection / union > max_iou:
            return True
    return False


def to_detector_line(box: tuple[int, int, int, int], image_size: tuple[int, int]) -> str:
    width, height = image_size
    x1, y1, x2, y2 = box
    cx = ((x1 + x2) / 2) / width
    cy = ((y1 + y2) / 2) / height
    bw = (x2 - x1) / width
    bh = (y2 - y1) / height
    return f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def generate_sample(
    tray_path: Path,
    food_paths: list[Path],
    min_items: int,
    max_items: int,
    min_scale: float,
    max_scale: float,
):
    Image, _, _, ImageOps = require_pillow()
    with Image.open(tray_path) as tray:
        canvas = ImageOps.exif_transpose(tray).convert("RGB")

    width, height = canvas.size
    boxes: list[tuple[int, int, int, int]] = []
    labels: list[str] = []
    item_count = random.randint(min_items, max_items)

    for food_path in random.choices(food_paths, k=item_count):
        food, mask = transform_food(food_path, min_scale, max_scale, canvas.size)
        if food.width >= width or food.height >= height:
            continue

        box = None
        for _ in range(80):
            x1 = random.randint(0, width - food.width)
            y1 = random.randint(0, height - food.height)
            candidate = (x1, y1, x1 + food.width, y1 + food.height)
            if not overlaps(candidate, boxes, max_iou=0.35):
                box = candidate
                break
        if box is None:
            continue

        canvas.paste(food, (box[0], box[1]), mask)
        boxes.append(box)
        labels.append(to_detector_line(box, canvas.size))

    return canvas, labels


def write_data_yaml(output_dir: Path) -> None:
    content = "\n".join(
        [
            f"path: {output_dir.as_posix()}",
            "train: images/train",
            "val: images/val",
            "names:",
            "  0: food_item",
            "",
        ]
    )
    (output_dir / "data.yaml").write_text(content, encoding="utf-8")


def prepare_output(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)
    for split in ["train", "val"]:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)


def main() -> None:
    configure_utf8_stdout()
    parser = argparse.ArgumentParser(description="Generate synthetic one-class detector tray data.")
    parser.add_argument("--food-root", default="data/raw/food_classes")
    parser.add_argument("--tray-root", default="data/raw/empty_trays")
    parser.add_argument("--output", default="data/generated/detector_synthetic")
    parser.add_argument("--samples", type=int, default=800)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--min-items", type=int, default=4)
    parser.add_argument("--max-items", type=int, default=6)
    parser.add_argument("--min-scale", type=float, default=0.20)
    parser.add_argument("--max-scale", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    food_root = project_path(args.food_root)
    tray_root = project_path(args.tray_root)
    output_dir = project_path(args.output)

    class_names = load_classes()
    food_paths = collect_food_images(food_root, class_names)
    tray_paths = [path for path in iter_images(tray_root) if path.suffix.lower() in IMAGE_EXTENSIONS]

    if not food_paths:
        raise SystemExit(f"No food images found in {food_root}")
    if not tray_paths:
        raise SystemExit(
            f"No empty tray images found in {tray_root}. "
            "Add top-down empty tray photos before generating detector data."
        )

    prepare_output(output_dir, overwrite=args.overwrite)
    val_count = int(args.samples * args.val_ratio)

    for index in range(args.samples):
        split = "val" if index < val_count else "train"
        tray_path = random.choice(tray_paths)
        image, labels = generate_sample(
            tray_path=tray_path,
            food_paths=food_paths,
            min_items=args.min_items,
            max_items=args.max_items,
            min_scale=args.min_scale,
            max_scale=args.max_scale,
        )
        stem = f"synthetic_{index:05d}"
        image.save(output_dir / "images" / split / f"{stem}.jpg", quality=95)
        (output_dir / "labels" / split / f"{stem}.txt").write_text(
            "\n".join(labels) + "\n",
            encoding="utf-8",
        )

    write_data_yaml(output_dir)
    print(f"Generated {args.samples} synthetic detector samples at {output_dir}")
    print(f"detector data config: {output_dir / 'data.yaml'}")


if __name__ == "__main__":
    main()
