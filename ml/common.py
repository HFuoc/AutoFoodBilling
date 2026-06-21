from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def project_path(*parts: str | Path) -> Path:
    return PROJECT_ROOT.joinpath(*parts)


def load_json(path: str | Path) -> Any:
    resolved = project_path(path) if not Path(path).is_absolute() else Path(path)
    with resolved.open("r", encoding="utf-8-sig") as file:
        return json.load(file)


def load_classes(path: str | Path = "configs/classes.json") -> list[str]:
    payload = load_json(path)
    return list(payload["classes"])


def iter_images(root: str | Path) -> list[Path]:
    resolved = project_path(root) if not Path(root).is_absolute() else Path(root)
    if not resolved.exists():
        return []
    return sorted(
        path
        for path in resolved.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def require_dependency(module_name: str, install_hint: str) -> None:
    try:
        __import__(module_name)
    except ImportError as exc:
        raise SystemExit(
            f"Missing dependency: {module_name}. Install dependencies with: {install_hint}"
        ) from exc


def configure_utf8_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
