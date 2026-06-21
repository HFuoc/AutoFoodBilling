from __future__ import annotations

import hashlib
import sys
from collections import defaultdict
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from ml.common import configure_utf8_stdout, iter_images, load_classes, project_path


def file_sha1(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    configure_utf8_stdout()
    root = project_path("data/raw/food_classes")
    class_names = load_classes()
    all_hashes: list[str] = []
    rows = []

    for class_name in class_names:
        class_dir = root / class_name
        images = iter_images(class_dir)
        hashes = [file_sha1(path) for path in images]
        all_hashes.extend(hashes)
        rows.append((class_name, len(images), len(set(hashes))))

    duplicate_groups = defaultdict(int)
    for image_hash in all_hashes:
        duplicate_groups[image_hash] += 1

    print("Class counts")
    print("------------")
    for class_name, total, unique in rows:
        print(f"{class_name}: total={total} unique={unique}")

    print()
    print(f"Total images: {len(all_hashes)}")
    print(f"Unique hashes: {len(set(all_hashes))}")
    print(f"Duplicate files: {len(all_hashes) - len(set(all_hashes))}")
    print(f"Duplicate groups: {sum(1 for count in duplicate_groups.values() if count > 1)}")


if __name__ == "__main__":
    main()
