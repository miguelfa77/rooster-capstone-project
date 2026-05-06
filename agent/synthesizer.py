"""Primitive response synthesizer for Rooster v2 Phase E."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field

from agent.config import (
    MAX_PRIMITIVES,
    REASONING_SYNTHESIZER,
    SYNTH_RESULT_SAMPLE_ROWS,
    SYNTH_TABLE_FALLBACK_ROWS,
    SYNTHESIZER_MAX_OUTPUT_TOKENS,
    SYNTHESIZER_MODEL_DEFAULT,
)
from agent.responses_api import (
    get_openai_client,
    parse_strict_response,
    reasoning_param_for_model,
    supports_temperature,
)
from agent.stage_logging import log_stage

_LOG = logging.getLogger("rooster.synthesizer")

PrimitiveKind = Literal["text", "kpi", "table", "code", "composite"]


class StrictBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


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


class CompositePrimitive(StrictBaseModel):
    kind: Literal["composite"] = "composite"
    heading: str | None = None
    blocks: list["PrimitiveBlock"] = Field(default_factory=list)


PrimitiveBlock = (
    TextPrimitive
    | KpiPrimitive
    | TablePrimitive
    | CodePrimitive
    | CompositePrimitive
)


class SynthesizedResponse(StrictBaseModel):
    primitives: list[PrimitiveBlock] = Field(default_factory=list)
    follow_ups: list[str] = Field(default_factory=list)


CompositePrimitive.model_rebuild()


_HARD_CONSTRAINTS = """HARD CONSTRAINTS — these override all other instructions:

1. HONOUR THE USER'S OUTPUT REQUEST.
   Read the USER_MESSAGE carefully. If the user requested a specific output format — in any
   phrasing, in Spanish or English — you must include the corresponding primitive in your response.

   Use your language understanding. Examples of what this means in practice, though this list is
   not exhaustive:

   Any request to see data on a map, geographically, spatially, "sobre el plano", "en el mapa",
   "visualización geográfica", "distribución por zonas", or similar → must include a code primitive
   with visualization_type "folium".

   Any request for a scatter plot, scatter chart, "gráfico de dispersión", "nube de puntos",
   "x contra y", "correlación entre", or similar → must include a code primitive with
   visualization_type "plotly" that creates a scatter plot.

   Any request for a table, "en tabla", "listado", "dame los datos", "muéstramelo ordenado",
   or similar → must include a table primitive.

   Any request for a quick number, "solo el número", "cuánto es", "dime solo",
   "el dato concreto", or similar → respond with a single kpi primitive and at most one short
   text primitive. Nothing else.

   Any request for a report, memo, "informe", "análisis completo", "resumen detallado",
   or similar → respond with a composite primitive with structured sections.

   Any request for a bar chart, ranking chart, "ranking visual", "ordenados en gráfico",
   or similar → must include a code primitive with visualization_type "plotly" that creates a bar chart.

   Any request for a trend, evolution, "cómo ha evolucionado", "a lo largo del tiempo",
   or similar → must include a code primitive with visualization_type "plotly" that creates a line chart.

   You may add additional primitives around the required one if they add genuine value.
   You may never omit the required one.

   If the user made no explicit format request, choose the primitives that best answer the
   question. Use your judgement.

2. NEVER USE RAW FIELD NAMES.
   Internal database identifiers must never appear in text primitives. Always use the Spanish display name.

   yield_pct → "rentabilidad bruta de alquiler"
   median_sale → "precio mediano de venta"
   median_alquiler → "precio mediano de alquiler"
   venta_count → "anuncios de venta"
   alquiler_count → "anuncios de alquiler"
   investment_score → "puntuación de inversión"
   tourism_pressure → "presión turística"
   transit_stop_count → "paradas de transporte cercanas"
   data_confidence → describe inline as "muestra reducida" for low; do not mention the field
   eur_per_sqm → "precio por metro cuadrado"
   value → "puntuación de inversión" when it represents that metric

3. CONFIDENCE IS ALWAYS INLINE.
   If a neighborhood has data_confidence "low", write "muestra reducida" immediately after the
   neighborhood's name or value in the same sentence.

   CORRECT: "Sant Isidre (7,5 %, muestra reducida) y Els Orriols (7,6 %)..."
   WRONG: "Sant Isidre (7,5 %) y Els Orriols (7,6 %)... Ojo: Sant Isidre tiene muestra reducida."

   Never write a trailing caveat paragraph about sample size.
"""


SYNTHESIZER_INSTRUCTIONS = """You are Rooster's Spanish real-estate analyst for Valencia.

Return ONLY JSON matching the provided schema. Compose the answer from the five allowed primitives:
text, kpi, table, code, composite.

Hard rules:
- Never use raw database field names in text. Use natural Spanish names from resolved_intent where available.
- Number formatting: Spanish style, e.g. 188.950 €, 7,2 %, counts as integers.
- Data confidence belongs inline next to the named item, not as a generic trailing caveat.
- Honor explicit output-format requests in USER_MESSAGE.
- Visuals must serve the answer. Do not emit default maps/charts just because data exists.
- Maximum six top-level primitives.

Primitive guidance:
- text: Spanish prose, concise. Use for lead, analysis, recommendation, caveat.
- kpi: one headline number, especially lookup answers.
- table: comparison/ranking/listing rows. rows_json must be a valid JSON array copied from agent_results.
- code: use for any visualization. Always return complete executable Python with fallback_kind="table".
- composite: only for longer memo-style answers.

Worked examples:
- "precio mediano en Russafa" -> kpi(label="Precio mediano de venta", value="...", unit="€") and a short text if needed.
- "scatter rentabilidad vs precio" -> text lead, code(visualization_type=plotly, code prints fig.to_json()).
- "compara Russafa y Benimaclet" -> text lead, table with both barrios and relevant metrics.
- "buena zona" -> text explaining the resolved concept, table or bar chart only if useful.

When producing visualizations, write complete executable Python code.

For Plotly charts:
- Import plotly.express or plotly.graph_objects as needed
- Build the figure from df (a pandas DataFrame of the executor results)
- Use exact column names from the data (check cited_data for column names)
- End with: print(fig.to_json())

For Folium maps (when user asks for geographic/map output):
- Import folium and geopandas
- For marker maps: build from df['lat'] and df['lng'] columns
- For choropleth maps: load geometry with:
  gdf = gpd.read_file('/data/valencia_barrios.geojson')
  then merge with df on neighborhood name
- End with: print(m.get_root().render())

The code primitive has a fallback_kind field. Set it to "table" always.
If the sandbox returns an error, the renderer will fall back to a table of raw data automatically.
Column names to use: write code referencing the exact column names present in the executor results.
The metrics registry canonical names are the column names (e.g. gross_rental_yield_pct, median_sale, etc.).
Do not invent column names.
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
    payload = {
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
        out, _response = parse_strict_response(
            get_openai_client(timeout_sec),
            SynthesizedResponse,
            **kwargs,
        )
        if isinstance(out, SynthesizedResponse):
            sanitized = _sanitize_response(out)
            response = sanitized
            _LOG.info("synthesized_response=%s", response.model_dump_json())
            log_stage(
                "synthesizer",
                "primitive_response",
                primitive_kinds=[p.kind for p in response.primitives],
                follow_up_count=len(response.follow_ups),
                correction_retry=bool(correction_block),
            )
            return response
    except Exception as exc:
        _LOG.exception("primitive_synthesis_failed: %s", exc)
    return _sanitize_response(_fallback_response(agent_results))


def synthesized_text(synthesized: SynthesizedResponse) -> str:
    parts: list[str] = []
    for primitive in synthesized.primitives:
        if isinstance(primitive, TextPrimitive) and primitive.content.strip():
            parts.append(primitive.content.strip())
        elif isinstance(primitive, KpiPrimitive):
            unit = f" {primitive.unit}" if primitive.unit else ""
            parts.append(f"{primitive.label}: {primitive.value}{unit}".strip())
    return "\n\n".join(parts)
