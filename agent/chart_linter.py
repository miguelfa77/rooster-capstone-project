"""Pure-Python chart primitive linter."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field

from agent.config import (
    CHART_MIN_HISTOGRAM_ROWS,
    CHART_MIN_LINE_ROWS,
    CHART_MIN_SCATTER_POINTS,
)
from agent.synthesizer import ChartPrimitive, SynthesizedResponse
from agent.stage_logging import log_stage


class LintResult(BaseModel):
    response: SynthesizedResponse
    modifications: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


def _rows(data_json: str) -> list[dict[str, Any]]:
    try:
        data = json.loads(data_json or "[]")
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    return [row for row in data if isinstance(row, dict)]


def _has_field(rows: list[dict[str, Any]], field: str | None) -> bool:
    if not field:
        return True
    return any(field in row and row.get(field) is not None for row in rows)


def _numeric(rows: list[dict[str, Any]], field: str | None) -> bool:
    if not field:
        return True
    seen = False
    for row in rows:
        if field not in row or row.get(field) is None:
            continue
        seen = True
        try:
            float(row[field])
        except (TypeError, ValueError):
            return False
    return seen


def _lint_chart(chart: ChartPrimitive) -> tuple[ChartPrimitive | None, list[str], list[str]]:
    rows = _rows(chart.data_json)
    modifications: list[str] = []
    errors: list[str] = []
    spec = chart.spec

    if spec.type in {"scatter", "bar", "line"}:
        if not spec.x or not spec.y:
            errors.append(f"{spec.type}: missing x/y encoding")
        if spec.x and not _has_field(rows, spec.x.field):
            errors.append(f"{spec.type}: missing x field {spec.x.field}")
        if spec.y and not _has_field(rows, spec.y.field):
            errors.append(f"{spec.type}: missing y field {spec.y.field}")
    if spec.type == "histogram":
        if not spec.x:
            errors.append("histogram: missing x encoding")
        elif not _has_field(rows, spec.x.field):
            errors.append(f"histogram: missing x field {spec.x.field}")
    if spec.type == "scatter":
        if len(rows) < CHART_MIN_SCATTER_POINTS:
            errors.append(f"scatter: needs at least {CHART_MIN_SCATTER_POINTS} points")
        if spec.x and not _numeric(rows, spec.x.field):
            errors.append(f"scatter: x field is not numeric: {spec.x.field}")
        if spec.y and not _numeric(rows, spec.y.field):
            errors.append(f"scatter: y field is not numeric: {spec.y.field}")
    if spec.type == "histogram" and len(rows) < CHART_MIN_HISTOGRAM_ROWS:
        errors.append(f"histogram: needs at least {CHART_MIN_HISTOGRAM_ROWS} rows")
    if spec.type == "line" and len(rows) < CHART_MIN_LINE_ROWS:
        errors.append(f"line: needs at least {CHART_MIN_LINE_ROWS} rows")
    if spec.color_field and not _has_field(rows, spec.color_field):
        modifications.append(f"dropped missing color field {spec.color_field}")
        spec = spec.model_copy(update={"color_field": None})
    if spec.size_field and not _has_field(rows, spec.size_field):
        modifications.append(f"dropped missing size field {spec.size_field}")
        spec = spec.model_copy(update={"size_field": None})
    if spec.x and spec.y and spec.x.field == spec.y.field:
        errors.append(f"{spec.type}: x and y cannot use the same field")
    if errors:
        return None, modifications, errors
    return chart.model_copy(update={"spec": spec}), modifications, errors


def lint_primitive_response(response: SynthesizedResponse) -> LintResult:
    kept = []
    modifications: list[str] = []
    errors: list[str] = []
    for primitive in response.primitives:
        if isinstance(primitive, ChartPrimitive):
            chart, mods, errs = _lint_chart(primitive)
            modifications.extend(mods)
            errors.extend(errs)
            if chart is not None:
                kept.append(chart)
            continue
        kept.append(primitive)
    result = LintResult(
        response=response.model_copy(update={"primitives": kept}),
        modifications=modifications,
        errors=errors,
    )
    log_stage(
        "chart_linter",
        "linted",
        chart_count=sum(1 for p in response.primitives if isinstance(p, ChartPrimitive)),
        kept_count=len(kept),
        modifications=modifications,
        errors=errors,
    )
    return result
