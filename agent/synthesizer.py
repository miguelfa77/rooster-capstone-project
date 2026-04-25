"""Structured answer synthesizer for Phase 3.5.

The synthesizer writes the primary answer first. Visuals are not chosen in
parallel; they are evidence hints attached to the specific prose claims they
support. The evidence selector may drop hints that fail deterministic rules.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Literal

from pydantic import BaseModel, Field

from agent.agent_pipeline import _build_results_summary_for_synth
from agent.responses_api import (
    get_openai_client,
    reasoning_param_for_model,
    supports_temperature,
)
from agent.template_registry import TemplateName, template_menu_for_prompt

_LOG = logging.getLogger("rooster.synthesizer")

SYNTHESIZER_MODEL = os.getenv("ROOSTER_SYNTHESIZER_MODEL", "gpt-5.5")

ClaimType = Literal[
    "ranking",
    "spatial_fact",
    "trend",
    "decomposition",
    "lookup",
    "comparison",
    "recommendation",
    "enumeration",
    "caveat",
    "no_data",
]


class DataReference(BaseModel):
    result_ref: str
    fields: list[str] = Field(default_factory=list)


class VisualEmphasis(BaseModel):
    metric: str | None = None
    neighborhoods: list[str] = Field(default_factory=list)


class VisualHint(BaseModel):
    template: TemplateName
    rationale: str
    data_subset: str
    emphasis: VisualEmphasis = Field(default_factory=VisualEmphasis)
    forced: bool = False


class Paragraph(BaseModel):
    text: str
    role: Literal["lead", "analysis", "recommendation", "caveat", "enumeration_intro"]
    claim_type: ClaimType
    cited_data: list[DataReference] = Field(default_factory=list)
    visual_hints: list[VisualHint] = Field(default_factory=list)


class SynthesizedResponse(BaseModel):
    paragraphs: list[Paragraph] = Field(default_factory=list)
    follow_ups: list[str] = Field(default_factory=list)


SYNTHESIZER_INSTRUCTIONS = """You are Rooster's Spanish real-estate analyst for Valencia.

Return ONLY JSON matching the provided schema. Do not output prose outside paragraphs[].text.

You write the primary answer. Visuals are optional evidence for specific claims, not decoration.
Use exact numbers from agent results. Never invent fields. If a field is missing, say so as a
caveat or no_data claim.

Prose discipline rules — these are non-negotiable:
1. NEVER use raw field names from the data in user-facing text. Translate them to natural Spanish:
   venta_count -> "anuncios de venta" or "muestra de venta";
   alquiler_count -> "anuncios de alquiler" or "muestra de alquiler";
   median_sale / median_venta / median_venta_price -> "precio mediano de venta";
   median_rent / median_alquiler / median_alquiler_price -> "precio mediano de alquiler";
   gross_rental_yield_pct / yield_pct -> "rentabilidad bruta de alquiler";
   investment_score -> "puntuación de inversión";
   tourism_pressure High -> "presión turística alta";
   tourism_pressure Medium/Moderate -> "presión turística moderada";
   tourism_pressure Low -> "presión turística baja";
   transit_stop_count -> "paradas de transporte";
   avg_dist_to_stop_m -> "distancia media a una parada";
   data_confidence low -> "muestra reducida" or "datos limitados".
2. Number formatting:
   - Prices: whole euros with Spanish thousands separator. 188950.0 -> "188.950 €"; 1500000 -> "1.500.000 €" or "1,5 M€".
   - Percentages and yields: one decimal and Spanish comma. 7.234 -> "7,2 %".
   - Counts: integers, no decimal point.
   - Distances: nearest 50 m for near/comparison phrasing; exact integer for specific facts.
3. Confidence handling is inline, never a trailing apology. When a cited neighborhood has
   data_confidence "low", mention it next to the neighborhood the first time it appears:
   "Sant Isidre (6,5 %, con muestra reducida)".
4. Do not apologize for filters that upstream tools applied. If a barrio is absent because it
   failed sample-size filters, do not speculate about it.
5. Do not introduce caveats the data does not warrant. Asking-price methodology caveats are
   appropriate when yields are central. Sample-size caveats are appropriate only when at least
   one cited row has data_confidence "low".
6. Tourism-pressure caveats are priority-aware. Only mention tourism pressure as a caveat if
   session_memory.user_profile.inferred_priorities.tourism_avoidance > 0 OR the current user
   message explicitly mentions turismo, VUT, alquiler vacacional, presión turística,
   masificación turística, Airbnb, or short-term rentals. Otherwise, do not add tourism caveats.

Visual hint discipline:
- Hints are suggestions. Hint a template only when it helps understand this paragraph's claim.
- Most paragraphs should have zero hints.
- ranking, spatial_fact, decomposition, comparison, and trend claims may have one hint.
- lookup and caveat claims usually have no hints.
- Use only template names from the closed menu.
- Make emphasis.metric match the metric named in the paragraph.

Worked example: ranking query
Input: "Which neighborhoods have the highest rental yield?"
Desired shape: lead paragraph with claim_type ranking, cited fields name/gross_rental_yield_pct,
one ranking_bars hint emphasizing gross_rental_yield_pct; optional spatial_fact paragraph with
choropleth_focus only if the result supports a spatial pattern; caveat paragraph with no hints.

Worked example: prose discipline
Input data fields include:
Sant Marcel.li: median_sale=188950.0, gross_rental_yield_pct=7.24, venta_count=12, alquiler_count=14, data_confidence=strong
Sant Isidre: median_sale=335000.0, gross_rental_yield_pct=6.52, venta_count=3, alquiler_count=3, data_confidence=low
Correct paragraph text:
"Como referencia de precios, Sant Marcel.li combina una rentabilidad bruta de alquiler del 7,2 % con un precio mediano de venta de 188.950 €, mientras que Sant Isidre alcanza el 6,5 % y 335.000 € de precio mediano de venta, aunque con muestra reducida."
Incorrect: "median_sale is 188950.0 €, venta_count is 12, data_confidence is low."

Worked example: comparison query
Input: "Compare Russafa and Benimaclet"
Desired shape: lead paragraph summarizing the key trade-off, claim_type comparison, cited fields
for both barrios, one comparison_matrix hint. Do not hint ranking_bars for two items.

Worked example: single lookup
Input: "What's the median price in Russafa?"
Desired shape: one lead lookup paragraph with the exact value and no visual_hints.

Worked example: investment memo recommendation
Input: "Where should I invest if I care about yield but want to avoid tourist pressure?"
Desired shape: recommendation lead using session priorities, comparison paragraph with one
comparison_matrix or scatter_2d hint, possible decomposition paragraph with score_breakdown if
investment_score is discussed, caveat paragraph with no hint. Never more than necessary.
"""


def _fallback_response(agent_results: list[dict[str, Any]]) -> SynthesizedResponse:
    summaries = _build_results_summary_for_synth(agent_results)
    if not summaries:
        return SynthesizedResponse(
            paragraphs=[
                Paragraph(
                    text="No tengo suficientes datos para responder con seguridad.",
                    role="caveat",
                    claim_type="no_data",
                )
            ],
            follow_ups=[],
        )
    first = summaries[0]
    tool = first.get("tool") or "consulta"
    n = first.get("row_count") or 0
    return SynthesizedResponse(
        paragraphs=[
            Paragraph(
                text=f"He encontrado {n} filas relevantes con {tool}.",
                role="lead",
                claim_type="lookup" if n else "no_data",
            )
        ],
        follow_ups=[],
    )


_FIELD_TEXT_REPLACEMENTS: dict[str, str] = {
    "tourism_pressure: High": "presión turística alta",
    "tourism_pressure: Medium": "presión turística moderada",
    "tourism_pressure: Moderate": "presión turística moderada",
    "tourism_pressure: Low": "presión turística baja",
    "data_confidence: low": "muestra reducida",
    "data_confidence low": "muestra reducida",
    "venta_count": "muestra de venta",
    "alquiler_count": "muestra de alquiler",
    "median_sale": "precio mediano de venta",
    "median_venta": "precio mediano de venta",
    "median_venta_price": "precio mediano de venta",
    "median_rent": "precio mediano de alquiler",
    "median_alquiler": "precio mediano de alquiler",
    "median_alquiler_price": "precio mediano de alquiler",
    "gross_rental_yield_pct": "rentabilidad bruta de alquiler",
    "yield_pct": "rentabilidad bruta de alquiler",
    "investment_score": "puntuación de inversión",
    "tourism_pressure": "presión turística",
    "transit_stop_count": "paradas de transporte",
    "avg_dist_to_stop_m": "distancia media a una parada",
    "data_confidence": "confianza de los datos",
}


def _sanitize_user_text(text: str) -> str:
    out = str(text or "")
    for raw, natural in _FIELD_TEXT_REPLACEMENTS.items():
        out = out.replace(raw, natural)
    return out


def _sanitize_response(response: SynthesizedResponse) -> SynthesizedResponse:
    return response.model_copy(
        update={
            "paragraphs": [
                p.model_copy(update={"text": _sanitize_user_text(p.text)})
                for p in response.paragraphs
            ]
        }
    )


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
    """Generate structured Spanish prose with claim-level visual hints."""
    if not (os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_KEY")):
        return _fallback_response(agent_results)

    client = get_openai_client(timeout_sec)
    result_summaries = _build_results_summary_for_synth(agent_results)
    payload = {
        "user_message": user_message,
        "resolved_intent": resolved_intent or {},
        "session_memory": session_memory or {},
        "template_menu": template_menu_for_prompt(),
        "agent_results": result_summaries,
    }
    model_name = model or SYNTHESIZER_MODEL
    kwargs: dict[str, Any] = {
        "model": model_name,
        "instructions": SYNTHESIZER_INSTRUCTIONS,
        "input": json.dumps(payload, ensure_ascii=False, default=str),
        "text_format": SynthesizedResponse,
        "max_output_tokens": 1800,
    }
    if supports_temperature(model_name):
        kwargs["temperature"] = 0.2
    if prompt_cache_key:
        kwargs["prompt_cache_key"] = prompt_cache_key
    rpar = reasoning_param_for_model(model_name, "low")
    if rpar is not None:
        kwargs["reasoning"] = rpar

    try:
        parsed = client.responses.parse(**kwargs)
        out = parsed.output_parsed
        if isinstance(out, SynthesizedResponse):
            sanitized = _sanitize_response(out)
            _LOG.info("synthesized_response=%s", sanitized.model_dump_json())
            return sanitized
    except Exception as exc:
        _LOG.exception("structured_synthesis_failed: %s", exc)
    return _sanitize_response(_fallback_response(agent_results))


def synthesized_text(synthesized: SynthesizedResponse) -> str:
    return "\n\n".join(p.text.strip() for p in synthesized.paragraphs if p.text.strip())
