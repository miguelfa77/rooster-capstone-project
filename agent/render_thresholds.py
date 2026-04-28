"""Data-confidence helpers backed by central Rooster config."""

from __future__ import annotations

from agent.config import (
    DATA_CONFIDENCE_ADEQUATE_MIN,
    DATA_CONFIDENCE_STRONG_MIN,
    MIN_ALQUILER_COUNT_DEFAULT,
    MIN_VENTA_COUNT_DEFAULT,
)


def data_confidence(venta_count: object, alquiler_count: object) -> str:
    try:
        venta = int(venta_count or 0)
    except (TypeError, ValueError):
        venta = 0
    try:
        alquiler = int(alquiler_count or 0)
    except (TypeError, ValueError):
        alquiler = 0
    if venta >= DATA_CONFIDENCE_STRONG_MIN and alquiler >= DATA_CONFIDENCE_STRONG_MIN:
        return "strong"
    if venta >= DATA_CONFIDENCE_ADEQUATE_MIN and alquiler >= DATA_CONFIDENCE_ADEQUATE_MIN:
        return "adequate"
    return "low"


def add_data_confidence(rows: list[dict]) -> list[dict]:
    out: list[dict] = []
    for row in rows:
        r = dict(row)
        r["data_confidence"] = data_confidence(
            r.get("venta_count"),
            r.get("alquiler_count"),
        )
        out.append(r)
    return out
