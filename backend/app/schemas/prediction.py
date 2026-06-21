from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class PredictionResponse(BaseModel):
    image: str
    detections: list[dict[str, Any]]
    bill: dict[str, Any]


class CellPredictionResponse(BaseModel):
    image: str
    cells: list[dict[str, Any]]
    bill: dict[str, Any]
