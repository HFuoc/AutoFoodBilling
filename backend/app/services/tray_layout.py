from __future__ import annotations

from pathlib import Path
from typing import Any


LANDSCAPE_TRAY_CELLS = [
    {"cell_id": "top_left", "box": (0.075, 0.080, 0.340, 0.390)},
    {"cell_id": "top_middle", "box": (0.390, 0.080, 0.640, 0.390)},
    {"cell_id": "top_right", "box": (0.680, 0.080, 0.940, 0.390)},
    {"cell_id": "bottom_left", "box": (0.060, 0.430, 0.420, 0.930)},
    {"cell_id": "bottom_right", "box": (0.450, 0.430, 0.940, 0.930)},
]


PORTRAIT_TRAY_CELLS = [
    {"cell_id": "top_left", "box": (0.070, 0.060, 0.620, 0.415)},
    {"cell_id": "top_right", "box": (0.660, 0.060, 0.940, 0.330)},
    {"cell_id": "middle_right", "box": (0.660, 0.365, 0.940, 0.655)},
    {"cell_id": "bottom_left", "box": (0.070, 0.555, 0.620, 0.940)},
    {"cell_id": "bottom_right", "box": (0.660, 0.720, 0.940, 0.940)},
]


def _missing_pillow() -> SystemExit:
    return SystemExit(
        "Missing dependency: pillow. Install dependencies with: "
        "pip install -r backend/requirements.txt"
    )


def detect_tray_bounds(image_path: Path) -> tuple[int, int, int, int] | None:
    try:
        import cv2
        import numpy as np
    except ImportError:
        return None

    image = cv2.imread(str(image_path))
    if image is None:
        return None

    height, width = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0, 0, 120]), np.array([180, 95, 255]))
    kernel = np.ones((9, 9), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[tuple[float, tuple[int, int, int, int]]] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area_ratio = (w * h) / max(1, width * height)
        aspect = w / max(1, h)
        if area_ratio < 0.35 or not 0.45 <= aspect <= 2.25:
            continue
        candidates.append((area_ratio, (x, y, x + w, y + h)))

    if not candidates:
        return None

    _, box = max(candidates, key=lambda item: item[0])
    x1, y1, x2, y2 = box
    pad_x = int(round((x2 - x1) * 0.015))
    pad_y = int(round((y2 - y1) * 0.015))
    return (
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(width, x2 + pad_x),
        min(height, y2 + pad_y),
    )


def standard_template_boxes(image_path: Path) -> list[dict[str, Any]]:
    try:
        from PIL import Image, ImageOps
    except ImportError as exc:
        raise _missing_pillow() from exc

    with Image.open(image_path) as image:
        image = ImageOps.exif_transpose(image)
        width, height = image.size

    tray_x1, tray_y1, tray_x2, tray_y2 = detect_tray_bounds(image_path) or (0, 0, width, height)
    tray_w = tray_x2 - tray_x1
    tray_h = tray_y2 - tray_y1
    template = PORTRAIT_TRAY_CELLS if tray_h > tray_w else LANDSCAPE_TRAY_CELLS

    boxes = []
    for cell in template:
        rx1, ry1, rx2, ry2 = cell["box"]
        boxes.append(
            {
                "cell_id": cell["cell_id"],
                "box": (
                    max(0, min(width, int(round(tray_x1 + rx1 * tray_w)))),
                    max(0, min(height, int(round(tray_y1 + ry1 * tray_h)))),
                    max(0, min(width, int(round(tray_x1 + rx2 * tray_w)))),
                    max(0, min(height, int(round(tray_y1 + ry2 * tray_h)))),
                ),
            }
        )
    return boxes


def crop_standard_template_cells(image_path: Path, output_dir: Path) -> list[dict[str, Any]]:
    try:
        from PIL import Image, ImageOps
    except ImportError as exc:
        raise _missing_pillow() from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    crops = []
    template_boxes = standard_template_boxes(image_path)
    with Image.open(image_path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        for index, cell in enumerate(template_boxes, start=1):
            box = cell["box"]
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            crop = image.crop(box)
            saved_path = output_dir / f"crop_{index:02d}.jpg"
            crop.save(saved_path, quality=95)
            crops.append(
                {
                    "index": index,
                    "box": box,
                    "confidence": 1.0,
                    "image": crop,
                    "path": str(saved_path),
                    "source": "standard_template",
                    "cell_id": cell["cell_id"],
                }
            )
    return crops
