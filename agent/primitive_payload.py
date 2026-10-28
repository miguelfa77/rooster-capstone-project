"""Make a synthesized primitive self-contained for API clients.

Descriptors get the rows they render (`data`), routed by column presence the same
way the Streamlit renderer does. `code` primitives are executed in the sandbox and
carry their output; if that fails they become a table of the data, as in v2.
"""

from __future__ import annotations

import json
from typing import Any

from agent.sandbox_client import call_sandbox


def _required_columns(primitive: dict) -> tuple[str, ...] | None:
    kind = primitive.get("kind")
    if kind == "choropleth":
        return ("geom", primitive.get("metric", ""))
    if kind == "bar":
        return (primitive.get("value_field", ""), primitive.get("label_field", ""))
    if kind in ("scatter", "line"):
        return (primitive.get("x_field", ""), primitive.get("y_field", ""))
    if kind == "point_map":
        return ("lat", "lng")
    return None


def _successful_rows(execution_results: list[dict]) -> list[list[dict]]:
    return [r["rows"] for r in execution_results or [] if r.get("success") and r.get("rows")]


def _rows_for_columns(execution_results: list[dict], required: tuple[str, ...]) -> list[dict]:
    for rows in _successful_rows(execution_results):
        if isinstance(rows[0], dict) and all(col in rows[0] for col in required):
            return rows
    return []


def _code_payload(primitive: dict, execution_results: list[dict]) -> dict:
    candidates = _successful_rows(execution_results)
    rows = candidates[0] if candidates else []
    if rows:
        result = call_sandbox(primitive.get("code", ""), rows)
        if result.get("success"):
            return {**primitive, "output_type": result.get("output_type"), "output": result.get("output") or ""}
    return {"kind": "table", "rows_json": json.dumps(rows, default=str), "columns": []}


def build_primitive_payload(primitive: dict[str, Any], execution_results: list[dict]) -> dict[str, Any]:
    kind = primitive.get("kind")
    if kind == "composite":
        blocks = [build_primitive_payload(b, execution_results) for b in primitive.get("blocks") or []]
        return {**primitive, "blocks": blocks}
    if kind == "code":
        return _code_payload(primitive, execution_results)
    required = _required_columns(primitive)
    if required is None:
        return primitive
    return {**primitive, "data": _rows_for_columns(execution_results, required)}
