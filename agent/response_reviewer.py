"""Reviewer 2: response correctness checks and final hard-constraint enforcement."""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from agent.config import (
    MAX_PRIMITIVES,
    REASONING_REVIEWER,
    REVIEWER_MAX_OUTPUT_TOKENS,
    REVIEWER_MODEL_DEFAULT,
)
from agent.responses_api import (
    get_openai_client,
    parse_strict_response,
    reasoning_param_for_model,
    supports_temperature,
)
from agent.semantic_layer.loader import load_registry
from agent.stage_logging import log_stage
from agent.synthesizer import (
    ChartPrimitive,
    CompositePrimitive,
    KpiPrimitive,
    MapPrimitive,
    PrimitiveBlock,
    SynthesizedResponse,
    TablePrimitive,
    TextPrimitive,
)

_LOG = logging.getLogger("rooster.response_reviewer")


class StrictBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ResponseReviewFailure(StrictBaseModel):
    type: Literal[
        "output_format_ignored",
        "raw_field_name",
        "confidence_trailing",
        "primitive_count",
        "implausible_followup",
    ]
    description: str
    suggested_correction: str


class ResponseReviewerVerdict(StrictBaseModel):
    verdict: Literal["pass", "fail"]
    failures: list[ResponseReviewFailure] = Field(default_factory=list)
    source: Literal["deterministic", "llm", "fallback"] = "llm"


RESPONSE_REVIEWER_INSTRUCTIONS = """You are a response quality reviewer for a real estate analytics assistant.

Your job is to check whether the synthesiser's output correctly answers
what the user asked for. You are NOT checking data correctness — assume
the data is correct. You are checking whether the response is well-formed,
honest, and matches the user's explicit requests.

You will receive:
- USER_MESSAGE: the original user message in Spanish
- PRIMITIVES: the list of primitives the synthesiser emitted, each with its type and key fields
- TEXT_CONTENT: the full text of all text primitives combined
- FORBIDDEN_FIELD_NAMES: list of raw database column names that must never appear in user-facing prose

Check:
1. OUTPUT FORMAT HONOURED:
   Read USER_MESSAGE carefully. Determine whether the user requested a specific output format.
   Use your language understanding — do not look for exact keyword matches.

   If the user requested a map or geographic visualisation in any phrasing: a primitive of type
   "map" must be present in PRIMITIVES.

   If the user requested a scatter plot or dispersion chart in any phrasing: a primitive of type
   "chart" with spec.type "scatter" must be present.

   If the user requested a table or ordered list in any phrasing: a primitive of type "table"
   must be present.

   If the user requested a quick number or single figure in any phrasing: only kpi and at most
   one text primitive should be present.

   If the user requested a report or detailed analysis in any phrasing: a "composite" primitive
   must be present.

   If the user made no explicit format request, the synthesiser had discretion. Do not fail on
   presentation grounds if no format was requested.

   When failing on this check, quote the exact phrase from USER_MESSAGE that constituted the
   format request, and state which primitive type is missing.
2. NO RAW FIELD NAMES: no forbidden field name may appear verbatim in TEXT_CONTENT.
3. CONFIDENCE INLINE: low-confidence caveats must be inline with the named item, not trailing.
4. PRIMITIVE COUNT APPROPRIATENESS: single-value lookups at most 2 primitives; comparisons or
   recommendations up to 4; more than 4 primitives is a failure; only text and no data primitive
   is a failure unless explicitly conversational.
5. FOLLOW-UPS PLAUSIBLE: follow_ups should contain 2-3 questions answerable by existing tools.

Return JSON only matching the schema. An output_format_ignored failure is always a hard failure.
"""


def field_name_map() -> dict[str, str]:
    """Map raw database identifiers to user-facing Spanish labels."""
    mapping: dict[str, str] = {
        "yield_pct": "rentabilidad bruta de alquiler",
        "median_sale": "precio mediano de venta",
        "median_alquiler": "precio mediano de alquiler",
        "median_venta_price": "precio mediano de venta",
        "median_alquiler_price": "precio mediano de alquiler",
        "venta_count": "anuncios de venta",
        "alquiler_count": "anuncios de alquiler",
        "investment_score": "puntuación de inversión",
        "tourism_pressure": "presión turística",
        "tourist_density_pct": "presión turística",
        "transit_stop_count": "paradas de transporte cercanas",
        "data_confidence": "muestra reducida",
        "eur_per_sqm": "precio por metro cuadrado",
        "value": "puntuación de inversión",
    }
    try:
        registry = load_registry()
    except Exception:
        return mapping
    for metric in registry.metrics:
        display = metric.display or metric.key.replace("_", " ")
        mapping[metric.key] = display
        column_name = str(metric.column or "").split(".")[-1]
        if column_name:
            mapping[column_name] = display
    return mapping


def forbidden_field_names() -> list[str]:
    return sorted(field_name_map())


def _text_blocks(primitives: list[PrimitiveBlock]) -> list[str]:
    parts: list[str] = []
    for primitive in primitives:
        if isinstance(primitive, TextPrimitive):
            if primitive.content.strip():
                parts.append(primitive.content.strip())
        elif isinstance(primitive, CompositePrimitive):
            parts.extend(_text_blocks(primitive.blocks))
    return parts


def _primitive_summary(primitives: list[PrimitiveBlock]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for primitive in primitives:
        item: dict[str, Any] = {"kind": primitive.kind}
        if isinstance(primitive, ChartPrimitive):
            item["chart_type"] = primitive.spec.type
        elif isinstance(primitive, MapPrimitive):
            item["layer_types"] = [layer.type for layer in primitive.layers]
        elif isinstance(primitive, TablePrimitive):
            item["columns"] = [col.field for col in primitive.columns]
        elif isinstance(primitive, CompositePrimitive):
            item["blocks"] = _primitive_summary(primitive.blocks)
        elif isinstance(primitive, TextPrimitive):
            item["role"] = primitive.role
        out.append(item)
    return out


def deterministic_response_review(
    *,
    user_message: str,
    response: SynthesizedResponse,
    field_names: list[str] | None = None,
) -> ResponseReviewerVerdict | None:
    failures: list[ResponseReviewFailure] = []
    text = "\n".join(_text_blocks(response.primitives))
    lowered_text = text.lower()
    for raw in field_names or forbidden_field_names():
        if raw and raw.lower() in lowered_text:
            failures.append(
                ResponseReviewFailure(
                    type="raw_field_name",
                    description=f"Raw field name '{raw}' appeared in user-facing text.",
                    suggested_correction=f"Replace '{raw}' with its Spanish display label.",
                )
            )
            break

    trailing_caveat = re.search(r"(?:^|\n\n)\s*(ojo|nota|ten en cuenta que)\b", text, flags=re.I)
    if trailing_caveat:
        failures.append(
            ResponseReviewFailure(
                type="confidence_trailing",
                description=f"Confidence caveat appears as trailing text: '{trailing_caveat.group(1)}'.",
                suggested_correction="Move low-confidence notes inline next to each named neighborhood.",
            )
        )

    if len(response.primitives) > MAX_PRIMITIVES:
        failures.append(
            ResponseReviewFailure(
                type="primitive_count",
                description=f"Response emitted more than {MAX_PRIMITIVES} top-level primitives.",
                suggested_correction=f"Trim the response to at most {MAX_PRIMITIVES} primitives.",
            )
        )
    has_data_primitive = any(
        isinstance(p, KpiPrimitive | TablePrimitive | ChartPrimitive | MapPrimitive)
        for p in response.primitives
    )
    if not has_data_primitive and not re.search(r"\b(hola|gracias|quien eres|qué puedes)\b", user_message, re.I):
        failures.append(
            ResponseReviewFailure(
                type="primitive_count",
                description="Response contains only text primitives for a data question.",
                suggested_correction="Add a KPI, table, chart, or map primitive using the available data.",
            )
        )

    if response.follow_ups and not (2 <= len(response.follow_ups) <= 3):
        failures.append(
            ResponseReviewFailure(
                type="implausible_followup",
                description=f"Response has {len(response.follow_ups)} follow-up suggestions.",
                suggested_correction="Provide 2-3 follow-up questions answerable by Rooster's tools.",
            )
        )

    if failures:
        return ResponseReviewerVerdict(verdict="fail", failures=failures, source="deterministic")
    return None


def review_synthesized_response(
    *,
    user_message: str,
    response: SynthesizedResponse,
    model: str | None = None,
    timeout_sec: float = 10.0,
    prompt_cache_key: str | None = None,
) -> ResponseReviewerVerdict:
    field_names = forbidden_field_names()
    precheck = deterministic_response_review(
        user_message=user_message,
        response=response,
        field_names=field_names,
    )
    if precheck is not None:
        log_stage("reviewer_2", "deterministic_fail", **precheck.model_dump())
        return precheck

    if not (os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_KEY")):
        verdict = ResponseReviewerVerdict(verdict="pass", source="fallback")
        log_stage("reviewer_2", "fallback_pass", **verdict.model_dump())
        return verdict

    payload = {
        "user_message": user_message,
        "primitives": _primitive_summary(response.primitives),
        "text_content": "\n".join(_text_blocks(response.primitives)),
        "forbidden_field_names": field_names,
        "follow_ups": response.follow_ups,
    }
    model_name = model or REVIEWER_MODEL_DEFAULT
    kwargs: dict[str, Any] = {
        "model": model_name,
        "instructions": RESPONSE_REVIEWER_INSTRUCTIONS,
        "input": json.dumps(payload, ensure_ascii=False, default=str),
        "text_format": ResponseReviewerVerdict,
        "max_output_tokens": REVIEWER_MAX_OUTPUT_TOKENS,
    }
    if supports_temperature(model_name):
        kwargs["temperature"] = 0
    if prompt_cache_key:
        kwargs["prompt_cache_key"] = prompt_cache_key
    rpar = reasoning_param_for_model(model_name, REASONING_REVIEWER)
    if rpar is not None:
        kwargs["reasoning"] = rpar

    try:
        out, _response = parse_strict_response(
            get_openai_client(timeout_sec),
            ResponseReviewerVerdict,
            **kwargs,
        )
        if isinstance(out, ResponseReviewerVerdict):
            out.source = "llm"
            log_stage("reviewer_2", "verdict", **out.model_dump())
            return out
    except Exception as exc:
        _LOG.exception("response_reviewer_failed: %s", exc)

    verdict = ResponseReviewerVerdict(verdict="pass", source="fallback")
    log_stage("reviewer_2", "fallback_pass", **verdict.model_dump())
    return verdict


def format_response_reviewer_correction(verdict: ResponseReviewerVerdict) -> str:
    lines = [
        "CORRECTION REQUIRED — your previous response failed review.",
        "Reviewer found the following problems. Fix all of them in this response.",
        "",
    ]
    for failure in verdict.failures:
        lines.append(f"- [{failure.type}] {failure.description}")
        lines.append(f"  Fix: {failure.suggested_correction}")
    lines.extend(
        [
            "",
            "The HARD CONSTRAINTS above still apply. Do not introduce new violations while fixing these.",
        ]
    )
    return "\n".join(lines)


def enforce_hard_constraints(
    response: SynthesizedResponse,
    *,
    mapping: dict[str, str] | None = None,
) -> SynthesizedResponse:
    primitives = list(response.primitives)
    fmap = mapping or field_name_map()
    cleaned: list[PrimitiveBlock] = []
    for primitive in primitives:
        if isinstance(primitive, TextPrimitive):
            content = primitive.content
            for raw, display in fmap.items():
                if raw in content:
                    content = content.replace(raw, display)
                    log_stage(
                        "reviewer_2",
                        "enforcement",
                        action="field_name_stripped",
                        field_name=raw,
                        display_name=display,
                    )
            cleaned.append(primitive.model_copy(update={"content": content}))
        else:
            cleaned.append(primitive)
    primitives = cleaned

    if len(primitives) > MAX_PRIMITIVES:
        original_count = len(primitives)
        primitives = primitives[:MAX_PRIMITIVES]
        log_stage(
            "reviewer_2",
            "enforcement",
            action="primitive_cap_applied",
            original_count=original_count,
            capped_count=MAX_PRIMITIVES,
        )
    return response.model_copy(update={"primitives": primitives})
