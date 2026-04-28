"""Primitive response synthesizer for Rooster v2 Phase E."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Literal, cast

from pydantic import BaseModel, Field

from agent.config import (
    MAX_PRIMITIVES,
    REASONING_SYNTHESIZER,
    SYNTH_RESULT_SAMPLE_ROWS,
    SYNTH_TABLE_FALLBACK_ROWS,
    SYNTHESIZER_MAX_OUTPUT_TOKENS,
    SYNTHESIZER_MODEL_DEFAULT,
)
from agent.responses_api import get_openai_client, reasoning_param_for_model, supports_temperature
from agent.stage_logging import log_stage

_LOG = logging.getLogger("rooster.synthesizer")

PrimitiveKind = Literal["text", "kpi", "table", "chart", "map", "composite"]


class TextPrimitive(BaseModel):
    kind: Literal["text"] = "text"
    content: str
    role: Literal["lead", "analysis", "recommendation", "caveat", "enumeration_intro"] = "analysis"
    cited_data: list[str] = Field(default_factory=list)


class KpiPrimitive(BaseModel):
    kind: Literal["kpi"] = "kpi"
    label: str
    value: str
    unit: str | None = None
    delta: str | None = None


class TableColumn(BaseModel):
    field: str
    header: str
    formatter: Literal["text", "currency", "percentage", "integer", "number", "link", "neighborhood-name"] = "text"
    alignment: Literal["left", "right", "center"] = "left"
    width_hint: Literal["narrow", "medium", "wide"] = "medium"


class TablePrimitive(BaseModel):
    kind: Literal["table"] = "table"
    rows_json: str = Field(default="[]", description="JSON array of row objects.")
    columns: list[TableColumn] = Field(default_factory=list)


class AxisSpec(BaseModel):
    field: str
    title: str
    type: Literal["quantitative", "nominal", "temporal"] = "quantitative"
    formatter: Literal["currency", "percentage", "integer", "number", "text", "date"] = "number"


class ChartSpec(BaseModel):
    type: Literal["bar", "line", "scatter", "histogram", "choropleth", "density"]
    x: AxisSpec | None = None
    y: AxisSpec | None = None
    color_field: str | None = None
    size_field: str | None = None
    label_field: str | None = None
    title: str | None = None


class ChartPrimitive(BaseModel):
    kind: Literal["chart"] = "chart"
    spec: ChartSpec
    data_json: str = Field(default="[]", description="JSON array of row objects.")


class MapLayer(BaseModel):
    type: Literal["markers", "choropleth", "polygons"]
    data_json: str = Field(default="[]", description="JSON array of row objects for this layer.")
    encoding: dict[str, str] = Field(default_factory=dict)


class MapPrimitive(BaseModel):
    kind: Literal["map"] = "map"
    layers: list[MapLayer] = Field(default_factory=list)
    focus: str | None = None


class CompositePrimitive(BaseModel):
    kind: Literal["composite"] = "composite"
    heading: str | None = None
    blocks: list["PrimitiveBlock"] = Field(default_factory=list)


PrimitiveBlock = (
    TextPrimitive
    | KpiPrimitive
    | TablePrimitive
    | ChartPrimitive
    | MapPrimitive
    | CompositePrimitive
)


class SynthesizedResponse(BaseModel):
    primitives: list[PrimitiveBlock] = Field(default_factory=list)
    follow_ups: list[str] = Field(default_factory=list)


CompositePrimitive.model_rebuild()


SYNTHESIZER_INSTRUCTIONS = """You are Rooster's Spanish real-estate analyst for Valencia.

Return ONLY JSON matching the provided schema. Compose the answer from the six allowed primitives:
text, kpi, table, chart, map, composite.

Hard rules:
- Never use raw database field names in text. Use natural Spanish names from resolved_intent where available.
- Number formatting: Spanish style, e.g. 188.950 €, 7,2 %, counts as integers.
- Data confidence belongs inline next to the named item, not as a generic trailing caveat.
- Honor presentation_hints. If the user asked for scatter, produce chart.type=scatter. If they asked for número rápido, produce one kpi plus at most one short text primitive.
- Visuals must serve the answer. Do not emit default maps/charts just because data exists.
- Maximum six top-level primitives.

Primitive guidance:
- text: Spanish prose, concise. Use for lead, analysis, recommendation, caveat.
- kpi: one headline number, especially lookup answers.
- table: comparison/ranking/listing rows. rows_json must be a valid JSON array copied from agent_results.
- chart: use spec encodings directly. data_json must contain the fields referenced by x/y/color/size/label.
- map: use only when spatial location is part of the answer. Put layer rows in layer.data_json.
- composite: only for longer memo-style answers.

Worked examples:
- "precio mediano en Russafa" -> kpi(label="Precio mediano de venta", value="...", unit="€") and a short text if needed.
- "scatter rentabilidad vs precio" -> text lead, chart(type=scatter, x=median_venta_price, y=gross_rental_yield_pct).
- "compara Russafa y Benimaclet" -> text lead, table with both barrios and relevant metrics.
- "buena zona" -> text explaining the resolved concept, table or bar chart only if useful.
"""


_FIELD_TEXT_REPLACEMENTS: dict[str, str] = {
    "venta_count": "muestra de venta",
    "alquiler_count": "muestra de alquiler",
    "median_venta_price": "precio mediano de venta",
    "median_alquiler_price": "precio mediano de alquiler",
    "gross_rental_yield_pct": "rentabilidad bruta de alquiler",
    "investment_score": "puntuación de inversión",
    "tourist_density_pct": "densidad de viviendas turísticas",
    "transit_stop_count": "paradas de transporte",
    "avg_dist_to_stop_m": "distancia media a una parada",
    "data_confidence": "confianza de los datos",
}


def _build_results_summary_for_synth(
    agent_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for result in agent_results or []:
        rows = result.get("rows") or []
        if result.get("success") and rows:
            sample = rows[:SYNTH_RESULT_SAMPLE_ROWS]
            columns = sorted({k for row in sample if isinstance(row, dict) for k in row})
            summaries.append(
                {
                    "tool": result.get("tool"),
                    "row_count": result.get("row_count"),
                    "columns": columns,
                    "sample": sample,
                    "params": result.get("params") or {},
                }
            )
        else:
            summaries.append(
                {
                    "tool": result.get("tool"),
                    "row_count": 0,
                    "note": result.get("error") or "sin filas",
                    "params": result.get("params") or {},
                }
            )
    return summaries


def _sanitize_text(text: str) -> str:
    out = str(text or "")
    for raw, natural in _FIELD_TEXT_REPLACEMENTS.items():
        out = out.replace(raw, natural)
    return out


def _json_rows(rows: list[dict[str, Any]], limit: int = 20) -> str:
    return json.dumps(rows[:limit], ensure_ascii=False, default=str)


def _first_success(agent_results: list[dict[str, Any]]) -> dict[str, Any] | None:
    return next(
        (r for r in agent_results or [] if r.get("success") and (r.get("rows") or [])),
        None,
    )


def _numeric_fields(row: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for key, val in row.items():
        if isinstance(val, bool):
            continue
        if isinstance(val, int | float):
            out.append(key)
            continue
        try:
            float(str(val))
        except (TypeError, ValueError):
            continue
        out.append(key)
    return out


def _columns_for_rows(rows: list[dict[str, Any]]) -> list[TableColumn]:
    if not rows:
        return []
    preferred = [
        "neighborhood_name",
        "name",
        "median_venta_price",
        "median_alquiler_price",
        "gross_rental_yield_pct",
        "investment_score",
        "tourist_density_pct",
        "transit_stop_count",
        "venta_count",
        "alquiler_count",
        "url",
    ]
    keys = [k for k in preferred if k in rows[0]]
    keys.extend(k for k in rows[0] if k not in keys)
    cols: list[TableColumn] = []
    for key in keys[:8]:
        formatter = "text"
        if key == "url":
            formatter = "link"
        elif key in {"neighborhood_name", "name"}:
            formatter = "neighborhood-name"
        elif "price" in key or "precio" in key:
            formatter = "currency"
        elif "pct" in key or "yield" in key:
            formatter = "percentage"
        elif key.endswith("_count") or key in {"rooms_int", "transit_stop_count"}:
            formatter = "integer"
        elif key in _numeric_fields(rows[0]):
            formatter = "number"
        cols.append(
            TableColumn(
                field=key,
                header=_FIELD_TEXT_REPLACEMENTS.get(key, key.replace("_", " ").title()),
                formatter=cast(Any, formatter),
                alignment="right" if formatter in {"currency", "percentage", "integer", "number"} else "left",
            )
        )
    return cols


def _fallback_response(agent_results: list[dict[str, Any]]) -> SynthesizedResponse:
    result = _first_success(agent_results)
    if result is None:
        return SynthesizedResponse(
            primitives=[
                TextPrimitive(
                    content="No tengo suficientes datos para responder con seguridad.",
                    role="caveat",
                )
            ],
            follow_ups=[],
        )
    rows = result.get("rows") or []
    tool = result.get("tool") or "consulta"
    primitives: list[PrimitiveBlock] = [
        TextPrimitive(content=f"He encontrado {len(rows)} filas relevantes con {tool}.", role="lead")
    ]
    if len(rows) == 1:
        numeric = _numeric_fields(rows[0])
        if numeric:
            field = numeric[0]
            primitives.append(
                KpiPrimitive(
                    label=_FIELD_TEXT_REPLACEMENTS.get(field, field.replace("_", " ").title()),
                    value=str(rows[0].get(field)),
                )
            )
    if rows:
        primitives.append(
            TablePrimitive(
                rows_json=_json_rows(rows, SYNTH_TABLE_FALLBACK_ROWS),
                columns=_columns_for_rows(rows),
            )
        )
    return SynthesizedResponse(primitives=primitives[:MAX_PRIMITIVES], follow_ups=[])


def _sanitize_response(response: SynthesizedResponse) -> SynthesizedResponse:
    sanitized: list[PrimitiveBlock] = []
    for primitive in response.primitives[:MAX_PRIMITIVES]:
        if isinstance(primitive, TextPrimitive):
            sanitized.append(primitive.model_copy(update={"content": _sanitize_text(primitive.content)}))
        else:
            sanitized.append(primitive)
    return response.model_copy(update={"primitives": sanitized})


def synthesize_response(
    user_message: str,
    *,
    resolved_intent: dict[str, Any] | None,
    agent_results: list[dict[str, Any]],
    session_memory: dict[str, Any] | None,
    model: str | None = None,
    timeout_sec: float = 45.0,
    prompt_cache_key: str | None = None,
) -> SynthesizedResponse:
    """Generate an ordered primitive response."""
    from agent.chart_linter import lint_primitive_response

    if not (os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_KEY")):
        return lint_primitive_response(_sanitize_response(_fallback_response(agent_results))).response

    result_summaries = _build_results_summary_for_synth(agent_results)
    payload = {
        "user_message": user_message,
        "resolved_intent": resolved_intent or {},
        "session_memory": session_memory or {},
        "agent_results": result_summaries,
    }
    model_name = model or SYNTHESIZER_MODEL_DEFAULT
    kwargs: dict[str, Any] = {
        "model": model_name,
        "instructions": SYNTHESIZER_INSTRUCTIONS,
        "input": json.dumps(payload, ensure_ascii=False, default=str),
        "text_format": SynthesizedResponse,
        "max_output_tokens": SYNTHESIZER_MAX_OUTPUT_TOKENS,
    }
    if supports_temperature(model_name):
        kwargs["temperature"] = 0.2
    if prompt_cache_key:
        kwargs["prompt_cache_key"] = prompt_cache_key
    rpar = reasoning_param_for_model(model_name, REASONING_SYNTHESIZER)
    if rpar is not None:
        kwargs["reasoning"] = rpar

    try:
        parsed = get_openai_client(timeout_sec).responses.parse(**kwargs)
        out = parsed.output_parsed
        if isinstance(out, SynthesizedResponse):
            sanitized = _sanitize_response(out)
            linted = lint_primitive_response(sanitized)
            if linted.errors:
                _LOG.warning("primitive_linter_errors=%s", linted.errors)
            _LOG.info("synthesized_response=%s", linted.response.model_dump_json())
            log_stage(
                "synthesizer",
                "primitive_response",
                primitive_kinds=[p.kind for p in linted.response.primitives],
                follow_up_count=len(linted.response.follow_ups),
                linter_errors=linted.errors,
                linter_modifications=linted.modifications,
            )
            return linted.response
    except Exception as exc:
        _LOG.exception("primitive_synthesis_failed: %s", exc)
    return lint_primitive_response(_sanitize_response(_fallback_response(agent_results))).response


def synthesized_text(synthesized: SynthesizedResponse) -> str:
    parts: list[str] = []
    for primitive in synthesized.primitives:
        if isinstance(primitive, TextPrimitive) and primitive.content.strip():
            parts.append(primitive.content.strip())
        elif isinstance(primitive, KpiPrimitive):
            unit = f" {primitive.unit}" if primitive.unit else ""
            parts.append(f"{primitive.label}: {primitive.value}{unit}".strip())
    return "\n\n".join(parts)
