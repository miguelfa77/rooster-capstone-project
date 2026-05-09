"""Primitive response synthesizer for Rooster v2 Phase E."""

from __future__ import annotations

import json
import logging
import os
from typing import Annotated, Any, Generator, Literal, Union, cast

from pydantic import Field

from agent.config import (
    MAX_PRIMITIVES,
    REASONING_SYNTHESIZER,
    SYNTH_RESULT_SAMPLE_ROWS,
    SYNTH_TABLE_FALLBACK_ROWS,
    SYNTHESIZER_MAX_OUTPUT_TOKENS,
    SYNTHESIZER_MODEL_DEFAULT,
)
from agent.responses_api import (
    StrictBaseModel,
    get_openai_client,
    parse_strict_response,
    reasoning_param_for_model,
    strict_json_schema_for_model,
    supports_temperature,
)
from agent.stage_logging import log_stage

_LOG = logging.getLogger("rooster.synthesizer")

PrimitiveKind = Literal["text", "kpi", "table", "code", "composite", "choropleth", "bar", "scatter", "line", "point_map"]


class TextPrimitive(StrictBaseModel):
    kind: Literal["text"] = "text"
    content: str
    role: Literal["lead", "analysis", "recommendation", "caveat", "enumeration_intro"] = "analysis"
    cited_data: list[str] = Field(default_factory=list)


class KpiPrimitive(StrictBaseModel):
    kind: Literal["kpi"] = "kpi"
    label: str
    value: str
    unit: str | None = None
    delta: str | None = None


class TableColumn(StrictBaseModel):
    field: str
    header: str
    formatter: Literal["text", "currency", "percentage", "integer", "number", "link", "neighborhood-name"] = "text"
    alignment: Literal["left", "right", "center"] = "left"
    width_hint: Literal["narrow", "medium", "wide"] = "medium"


class TablePrimitive(StrictBaseModel):
    kind: Literal["table"] = "table"
    rows_json: str = Field(default="[]", description="JSON array of row objects.")
    columns: list[TableColumn] = Field(default_factory=list)


class CodePrimitive(StrictBaseModel):
    kind: Literal["code"] = "code"
    visualization_type: Literal["plotly", "folium"]
    code: str
    fallback_kind: Literal["table"] = "table"


class TooltipField(StrictBaseModel):
    field: str
    label: str


class ChoroplethDescriptor(StrictBaseModel):
    kind: Literal["choropleth"] = "choropleth"
    metric: str
    metric_label: str
    colormap: Literal["YlOrRd", "YlGn", "Blues", "RdYlGn", "Purples"] = "YlOrRd"
    tooltip_fields: list[TooltipField] = Field(default_factory=list)
    tiles: str = "cartodbpositron"


class BarDescriptor(StrictBaseModel):
    kind: Literal["bar"] = "bar"
    value_field: str
    label_field: str
    value_label: str
    orientation: Literal["h", "v"] = "h"
    top_n: int = 15


class ScatterDescriptor(StrictBaseModel):
    kind: Literal["scatter"] = "scatter"
    x_field: str
    y_field: str
    x_label: str
    y_label: str
    label_field: str | None = None
    color_field: str | None = None


class LineDescriptor(StrictBaseModel):
    kind: Literal["line"] = "line"
    x_field: str
    y_field: str
    y_label: str
    color_field: str | None = None


class PointMapDescriptor(StrictBaseModel):
    kind: Literal["point_map"] = "point_map"
    color_field: str | None = None
    color_label: str | None = None
    popup_fields: list[TooltipField] = Field(default_factory=list)


# Non-recursive union — used inside CompositePrimitive.blocks to avoid circular schema.
# CompositePrimitive itself is intentionally excluded here.
InnerPrimitiveBlock = Annotated[
    Union[
        TextPrimitive,
        KpiPrimitive,
        TablePrimitive,
        CodePrimitive,
        ChoroplethDescriptor,
        BarDescriptor,
        ScatterDescriptor,
        LineDescriptor,
        PointMapDescriptor,
    ],
    Field(discriminator="kind"),
]


class CompositePrimitive(StrictBaseModel):
    kind: Literal["composite"] = "composite"
    heading: str | None = None
    blocks: list[InnerPrimitiveBlock] = Field(default_factory=list)


PrimitiveBlock = Annotated[
    Union[
        TextPrimitive,
        KpiPrimitive,
        TablePrimitive,
        CodePrimitive,
        ChoroplethDescriptor,
        BarDescriptor,
        ScatterDescriptor,
        LineDescriptor,
        PointMapDescriptor,
        CompositePrimitive,
    ],
    Field(discriminator="kind"),
]


class SynthesizedResponse(StrictBaseModel):
    primitives: list[PrimitiveBlock] = Field(default_factory=list)
    follow_ups: list[str] = Field(default_factory=list)


_HARD_CONSTRAINTS = """CORRECTION MODE: If the input contains a "correction_required" key, a previous code
primitive failed with that error. Rewrite only the code primitive to fix it. Do not change
text, kpi, table, or descriptor primitives.

HARD CONSTRAINTS — these override all other instructions:

1. HONOUR THE USER'S OUTPUT REQUEST using the right primitive:

   Neighborhood map colored by a metric ("mapa por barrios", "coropleta", "distribución geográfica"):
   → choropleth descriptor. Requires select_metrics with include_geometry:true.

   Map of individual listings/points ("pisa en el mapa", "dónde están", point markers):
   → point_map descriptor. Requires lat/lng columns.

   Scatter / correlation / "X vs Y" / "relación entre":
   → scatter descriptor.

   Bar chart / ranking / "top N" / "ordenados en gráfico":
   → bar descriptor.

   Trend / time series / "cómo ha evolucionado" / "a lo largo del tiempo":
   → line descriptor.

   Table / listing / "en tabla" / "dame los datos":
   → table primitive.

   Single number / KPI / "cuánto es" / "dime solo el dato":
   → kpi primitive.

   Report / memo / "informe completo":
   → composite primitive.

   Non-standard visualization not covered above:
   → code primitive (last resort only).

   You may never omit the required primitive. All field names in descriptors must exactly
   match column names from agent_results.

2. NEVER USE RAW FIELD NAMES in text primitives. Always use Spanish display names:
   gross_rental_yield_pct → "rentabilidad bruta de alquiler"
   median_venta_price → "precio mediano de venta"
   median_alquiler_price → "precio mediano de alquiler"
   venta_count → "anuncios de venta"
   alquiler_count → "anuncios de alquiler"
   investment_score → "puntuación de inversión"
   tourist_density_pct → "presión turística"
   transit_stop_count → "paradas de transporte cercanas"
   data_confidence → describe inline as "muestra reducida" for low; never mention the field name
   median_venta_eur_per_sqm → "precio por metro cuadrado"

3. CONFIDENCE IS ALWAYS INLINE.
   CORRECT: "Sant Isidre (7,5 %, muestra reducida) y Els Orriols (7,6 %)..."
   WRONG: trailing caveat paragraph about sample size.
"""


SYNTHESIZER_INSTRUCTIONS = """You are Rooster's Spanish real-estate analyst for Valencia.

Return ONLY JSON matching the provided schema. Hard rules:
- Never use raw field names in text primitives. Use Spanish display names.
- Number formatting: Spanish style — 188.950 €, 7,2 %, counts as integers.
- Data confidence inline, never as trailing caveat.
- Visuals must serve the answer. Do not emit charts when data does not support them.
- Maximum six top-level primitives.

VISUALIZATION DESCRIPTORS — use these for standard charts. They render via verified Python
templates with no code generation risk. All field names must exactly match agent_results columns.

choropleth — neighborhood polygon map colored by a metric:
  Requires geom column (select_metrics with include_geometry:true).
  {"kind":"choropleth","metric":"gross_rental_yield_pct","metric_label":"Rentabilidad bruta de alquiler (%)","colormap":"YlOrRd","tooltip_fields":[{"field":"neighborhood_name","label":"Barrio"},{"field":"gross_rental_yield_pct","label":"Rentabilidad bruta de alquiler"}]}

bar — ranking / top-N bar chart:
  {"kind":"bar","value_field":"gross_rental_yield_pct","label_field":"neighborhood_name","value_label":"Rentabilidad bruta de alquiler (%)","orientation":"h","top_n":15}

scatter — two-metric scatter / correlation:
  {"kind":"scatter","x_field":"gross_rental_yield_pct","y_field":"investment_score","x_label":"Rentabilidad bruta (%)","y_label":"Puntuación de inversión","label_field":"neighborhood_name"}

line — time series / trend:
  {"kind":"line","x_field":"time_point","y_field":"value","y_label":"Precio mediano de venta (€)","color_field":"neighborhood_name"}

point_map — marker map for individual listings (requires lat/lng columns):
  {"kind":"point_map","color_field":"eur_per_sqm","color_label":"€/m²","popup_fields":[{"field":"price_int","label":"Precio"},{"field":"neighborhood_name","label":"Barrio"}]}

OTHER PRIMITIVES:
- text: Spanish prose. Use for lead, analysis, recommendation.
- kpi: single headline number for lookup answers.
- table: comparison/ranking data. rows_json must be a valid JSON array from agent_results.
- composite: structured sections for memo/report answers.
- code: ONLY for non-standard visualizations not covered by descriptors above. When used,
  set fallback_kind="table" and write complete executable Python ending with print(output).

When multiple tools returned data (e.g., select_metrics + query_listings), emit a separate
descriptor for each dataset. Each descriptor's renderer automatically selects the result
whose columns match — you do not need to specify which result to use.

Worked examples:
- "mapa de barrios por yield" → text lead + choropleth descriptor
- "ranking de barrios por rentabilidad" → text lead + bar descriptor
- "correlación yield vs score" → text lead + scatter descriptor
- "precio mediano en Russafa" → kpi + short text
- "compara Russafa y Benimaclet" → text lead + table
- "mapa de barrios y listado de pisos baratos" → choropleth (select_metrics data) + point_map (query_listings data)
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
    correction_block: str | None = None,
) -> SynthesizedResponse:
    """Generate an ordered primitive response."""
    if not (os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_KEY")):
        return _sanitize_response(_fallback_response(agent_results))

    result_summaries = _build_results_summary_for_synth(agent_results)
    resolved = resolved_intent or {}
    input_payload: dict[str, Any] = {
        "user_message": user_message,
        "resolved_intent": resolved,
        "session_memory": session_memory or {},
        "agent_results": result_summaries,
    }
    if correction_block:
        input_payload = {
            "correction_required": correction_block,
            **input_payload,
        }
    instructions = "\n\n".join(
        [
            _HARD_CONSTRAINTS,
            SYNTHESIZER_INSTRUCTIONS,
        ]
    )
    model_name = model or SYNTHESIZER_MODEL_DEFAULT
    kwargs: dict[str, Any] = {
        "model": model_name,
        "instructions": instructions,
        "input": json.dumps(input_payload, ensure_ascii=False, default=str),
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
        out, _response = parse_strict_response(
            get_openai_client(timeout_sec),
            SynthesizedResponse,
            **kwargs,
        )
        if isinstance(out, SynthesizedResponse):
            sanitized = _sanitize_response(out)
            _LOG.info("synthesized_response=%s", sanitized.model_dump_json())
            log_stage(
                "synthesizer",
                "primitive_response",
                primitive_kinds=[p.kind for p in sanitized.primitives],
                follow_up_count=len(sanitized.follow_ups),
                correction_retry=bool(correction_block),
            )
            return sanitized
    except Exception as exc:
        _LOG.exception("primitive_synthesis_failed: %s", exc)
    return _sanitize_response(_fallback_response(agent_results))


class _ProseStreamer:
    """Extracts TextPrimitive content fields from a streaming JSON response."""

    def __init__(self) -> None:
        self._buf = ""
        self._yielding = False
        self._escape = False

    def feed(self, chunk: str) -> str:
        out: list[str] = []
        for ch in chunk:
            if not self._yielding:
                self._buf += ch
                if len(self._buf) > 15:
                    self._buf = self._buf[-15:]
                if self._buf.endswith('"content":"'):
                    self._yielding = True
                    self._buf = ""
            else:
                if self._escape:
                    self._escape = False
                    if ch == "n":
                        out.append("\n")
                    elif ch == "t":
                        out.append("\t")
                    elif ch in ('"', "\\", "/"):
                        out.append(ch)
                elif ch == "\\":
                    self._escape = True
                elif ch == '"':
                    self._yielding = False
                    out.append(" ")
                else:
                    out.append(ch)
        return "".join(out)


def stream_synthesize_response(
    user_message: str,
    *,
    resolved_intent: dict[str, Any] | None,
    agent_results: list[dict[str, Any]],
    session_memory: dict[str, Any] | None,
    model: str | None = None,
    timeout_sec: float = 45.0,
    prompt_cache_key: str | None = None,
    correction_block: str | None = None,
) -> Generator[str | SynthesizedResponse, None, None]:
    """Yields str prose deltas as the synthesis streams, then a final SynthesizedResponse."""
    if not (os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_KEY")):
        yield _sanitize_response(_fallback_response(agent_results))
        return

    input_payload: dict[str, Any] = {
        "user_message": user_message,
        "resolved_intent": resolved_intent or {},
        "session_memory": session_memory or {},
        "agent_results": _build_results_summary_for_synth(agent_results),
    }
    if correction_block:
        input_payload = {"correction_required": correction_block, **input_payload}

    instructions = "\n\n".join([_HARD_CONSTRAINTS, SYNTHESIZER_INSTRUCTIONS])
    model_name = model or SYNTHESIZER_MODEL_DEFAULT
    kwargs: dict[str, Any] = {
        "model": model_name,
        "instructions": instructions,
        "input": json.dumps(input_payload, ensure_ascii=False, default=str),
        "text": {
            "format": {
                "type": "json_schema",
                "name": SynthesizedResponse.__name__,
                "schema": strict_json_schema_for_model(SynthesizedResponse),
                "strict": True,
            }
        },
        "max_output_tokens": SYNTHESIZER_MAX_OUTPUT_TOKENS,
        "stream": True,
    }
    if supports_temperature(model_name):
        kwargs["temperature"] = 0.2
    if prompt_cache_key:
        kwargs["prompt_cache_key"] = prompt_cache_key
    rpar = reasoning_param_for_model(model_name, REASONING_SYNTHESIZER)
    if rpar is not None:
        kwargs["reasoning"] = rpar

    try:
        client = get_openai_client(timeout_sec)
        streamer = _ProseStreamer()
        full_text = ""
        for event in client.responses.create(**kwargs):
            if getattr(event, "type", "") == "response.output_text.delta":
                delta = getattr(event, "delta", "") or ""
                if delta:
                    full_text += delta
                    prose = streamer.feed(delta)
                    if prose:
                        yield prose
        try:
            result = SynthesizedResponse.model_validate(json.loads(full_text))
            log_stage(
                "synthesizer",
                "primitive_response",
                primitive_kinds=[p.kind for p in result.primitives],
                follow_up_count=len(result.follow_ups),
                correction_retry=bool(correction_block),
            )
            yield _sanitize_response(result)
        except Exception:
            yield _sanitize_response(_fallback_response(agent_results))
    except Exception as exc:
        _LOG.exception("stream_synthesis_failed: %s", exc)
        yield _sanitize_response(_fallback_response(agent_results))


def synthesized_text(synthesized: SynthesizedResponse) -> str:
    parts: list[str] = []
    for primitive in synthesized.primitives:
        if isinstance(primitive, TextPrimitive) and primitive.content.strip():
            parts.append(primitive.content.strip())
        elif isinstance(primitive, KpiPrimitive):
            unit = f" {primitive.unit}" if primitive.unit else ""
            parts.append(f"{primitive.label}: {primitive.value}{unit}".strip())
    return "\n\n".join(parts)
