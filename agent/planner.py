"""Unified data/conversation planner for Rooster v2 Phase C."""

from __future__ import annotations

import json
import logging
import os
import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field

from agent.config import (
    PLANNER_CONVERSATION_CONTEXT_CHARS,
    PLANNER_LAST_ASSISTANT_CONTEXT_CHARS,
    PLANNER_MAX_OUTPUT_TOKENS,
    PLANNER_MODEL_DEFAULT,
    PLANNER_SCHEMA_SNAPSHOT_CHARS,
    REASONING_PLANNER,
)
from agent.responses_api import (
    get_openai_client,
    reasoning_param_for_model,
    supports_temperature,
)
from agent.semantic_layer.models import ResolvedQuery
from agent.stage_logging import log_stage

_LOG = logging.getLogger("rooster.planner")


class PlannedToolCall(BaseModel):
    tool: str
    params_json: str = Field(
        default="{}",
        description="A JSON object string containing the tool parameters.",
    )
    rationale: str = ""


class PlannerDecision(BaseModel):
    route: Literal["data", "conversational"]
    conversational_response: str | None = None
    tool_calls: list[PlannedToolCall] = Field(default_factory=list)
    reasoning_summary: str = ""
    needs_more_data: bool = False


PLANNER_INSTRUCTIONS = """You are Rooster's unified planner for Valencia real estate.

Return ONLY JSON matching the schema. Choose exactly one route:
- conversational: greetings, thanks, meta questions, or clarification-style replies that need no database.
- data: analytical/listing/map questions that require tools.

The deterministic semantic resolver already ran. You MUST respect ResolvedQuery:
- Use resolved_metrics when selecting metrics.
- Use resolved_concepts as filters expressed through select_metrics where possible.
- Use resolved_heuristics as neighborhood/spatial constraints.
- Read the original user message for requested output shape: scatter/trend/map/table-style requests need data that can support that output.
- If unresolved_essential_terms is non-empty, route conversational and ask a concise Spanish clarification.

Prefer compositional tools:
- select_metrics for neighborhood rankings, comparisons, metric tables, and filtered analytical queries.
- compute_aggregate for city-wide summary statistics.
- temporal_series for trends over time.
- query_listings only for individual property listings.
- query_transit_stops, query_tourist_apartments, resolve_spatial_reference only when those specific records are needed.

Do not emit prose when route=data. Tool params must be concrete and validated-looking.
For every tool call, put arguments in params_json as a valid JSON object string, not markdown.
Spanish conversational responses only, max 3 sentences.
"""


def _fallback_decision(user_message: str, resolved_query: ResolvedQuery) -> PlannerDecision:
    if resolved_query.needs_clarification:
        terms = ", ".join(resolved_query.unresolved_essential_terms)
        return PlannerDecision(
            route="conversational",
            conversational_response=(
                "Necesito aclarar algunos términos antes de consultar los datos: "
                f"{terms}. ¿A qué te refieres exactamente?"
            ),
            reasoning_summary="semantic_clarification",
        )
    t = (user_message or "").strip().lower()
    if t in {"hola", "hi", "hey", "gracias", "thanks", "ok", "vale"}:
        return PlannerDecision(
            route="conversational",
            conversational_response="Hola. Soy Rooster, tu analista inmobiliario de Valencia. ¿Qué quieres analizar?",
            reasoning_summary="simple_conversation",
        )
    metrics = [m.key for m in resolved_query.resolved_metrics] or ["investment_score"]
    order_metric = metrics[0]
    return PlannerDecision(
        route="data",
        tool_calls=[
            PlannedToolCall(
                tool="select_metrics",
                params_json=json.dumps({
                    "metrics": metrics,
                    "group_by": ["neighborhood"],
                    "order_by": {"metric": order_metric, "direction": "desc"},
                    "limit": 10,
                }),
                rationale="fallback_select_metrics",
            )
        ],
        reasoning_summary="fallback_data_plan",
    )


def plan_query(
    user_message: str,
    *,
    resolved_query: ResolvedQuery,
    session_memory: dict[str, Any],
    live_schema_snapshot: str,
    conversation_context: str = "",
    last_assistant_context: str = "",
    correction_hint: str | None = None,
    previous_results: list[dict[str, Any]] | None = None,
    model: str | None = None,
    timeout_sec: float = 45.0,
    prompt_cache_key: str | None = None,
) -> tuple[PlannerDecision, str | None]:
    """Plan one loop step as structured JSON."""
    if not (os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_KEY")):
        return _fallback_decision(user_message, resolved_query), None

    payload = {
        "user_message": user_message,
        "resolved_query": resolved_query.model_dump(),
        "session_memory": session_memory,
        "live_schema_snapshot": live_schema_snapshot[:PLANNER_SCHEMA_SNAPSHOT_CHARS],
        "conversation_context": conversation_context[-PLANNER_CONVERSATION_CONTEXT_CHARS:],
        "last_assistant_context": last_assistant_context[-PLANNER_LAST_ASSISTANT_CONTEXT_CHARS:],
        "correction_hint": correction_hint or "",
        "previous_results": previous_results or [],
    }
    model_name = model or PLANNER_MODEL_DEFAULT
    kwargs: dict[str, Any] = {
        "model": model_name,
        "instructions": PLANNER_INSTRUCTIONS,
        "input": json.dumps(payload, ensure_ascii=False, default=str),
        "text_format": PlannerDecision,
        "max_output_tokens": PLANNER_MAX_OUTPUT_TOKENS,
    }
    if supports_temperature(model_name):
        kwargs["temperature"] = 0
    if prompt_cache_key:
        kwargs["prompt_cache_key"] = prompt_cache_key
    rpar = reasoning_param_for_model(model_name, REASONING_PLANNER)
    if rpar is not None:
        kwargs["reasoning"] = rpar

    try:
        client = get_openai_client(timeout_sec)
        parsed = client.responses.parse(**kwargs)
        out = parsed.output_parsed
        if isinstance(out, PlannerDecision):
            _LOG.info("planner_decision=%s", out.model_dump_json())
            log_stage(
                "planner",
                "llm_decision",
                route=out.route,
                tool_calls=[call.tool for call in out.tool_calls],
                needs_more_data=out.needs_more_data,
            )
            return out, getattr(parsed, "id", None)
    except Exception as exc:
        _LOG.exception("planner_failed: %s", exc)
    return _fallback_decision(user_message, resolved_query), f"fallback-{uuid.uuid4().hex[:12]}"


def planner_tool_calls_to_plan_calls(tool_calls: list[PlannedToolCall]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for call in tool_calls:
        try:
            params = json.loads(call.params_json or "{}")
        except json.JSONDecodeError:
            params = {}
        if not isinstance(params, dict):
            params = {}
        out.append(
            {
                "tool": call.tool,
                "params": params,
                "_tool_call_id": f"planner_{uuid.uuid4().hex[:12]}",
                "planner_rationale": call.rationale,
            }
        )
    return out
