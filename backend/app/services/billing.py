from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MENU_PATH = PROJECT_ROOT / "configs" / "menu.json"


def load_menu(menu_path: str | Path = DEFAULT_MENU_PATH) -> dict[str, Any]:
    with Path(menu_path).open("r", encoding="utf-8-sig") as file:
        return json.load(file)


def build_bill(labels: list[str], menu_path: str | Path = DEFAULT_MENU_PATH) -> dict[str, Any]:
    menu = load_menu(menu_path)
    items = menu["items"]
    currency = menu.get("currency", "VND")
    lines = []
    total = 0

    for index, label in enumerate(labels, start=1):
        menu_item = items.get(label)
        price = int(menu_item["price"]) if menu_item else 0
        total += price
        lines.append(
            {
                "index": index,
                "name": label,
                "price": price,
                "currency": currency,
                "known": menu_item is not None,
            }
        )

    return {"currency": currency, "items": lines, "total": total}


def format_bill(bill: dict[str, Any]) -> str:
    lines = ["Detected dishes:"]
    for item in bill["items"]:
        price = f"{item['price']:,}".replace(",", ".")
        lines.append(f"{item['index']}. {item['name']} - {price} {bill['currency']}")
    total = f"{bill['total']:,}".replace(",", ".")
    lines.append(f"Total: {total} {bill['currency']}")
    return "\n".join(lines)
