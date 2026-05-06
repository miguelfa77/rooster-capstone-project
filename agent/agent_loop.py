"""Bounded v2 Responses tool loop for Rooster."""

from __future__ import annotations

import logging
from typing import Any

from agent.config import (
    MAX_AGENT_STEPS_DEFAULT,
    MAX_AGENT_STEPS_RECOMMENDATION,
    SYNTHESIZER_MODEL_DEFAULT,
)
from agent.stage_logging import log_stage

MAX_STEPS_DEFAULT = MAX_AGENT_STEPS_DEFAULT
_LOG = logging.getLogger("rooster.agent_loop")


def compute_max_steps(resolved_intent: dict[str, Any] | None) -> int:
    if not resolved_intent:
        return MAX_STEPS_DEFAULT
    if resolved_intent.get("comparison_mode") == "find_best" or resolved_intent.get(
        "recommendation_requested"
    ):
        return max(MAX_STEPS_DEFAULT, MAX_AGENT_STEPS_RECOMMENDATION)
    return MAX_STEPS_DEFAULT


def run_agent_loop(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Backwards-compatible positional wrapper around ``run_agent_loop_pipeline``."""
    return run_agent_loop_pipeline(*args, **kwargs)


def run_agent_loop_pipeline(
    user_input: str,
    conversation_state: dict[str, Any],
    live_schema_context: str,
    static_schema: str,
    conversation_context: str,
    model: str,
    timeout_sec: float,
    engine,
    last_assistant_context: str = "",
    prompt_cache_key: str | None = None,
    previous_planner_response_id: str | None = None,
    preamble_callback=None,
) -> dict[str, Any]:
    """
    Unified planner loop: resolve semantics, route conversational/data, validate,
    execute, then optionally re-plan while budget remains.
    """
    from agent.agent_pipeline import (
        _build_results_summary_for_synth,
        _infer_combine_maps_from_tools,
        _infer_plan_neighborhood_resolved,
        execute_plan,
        format_validation_plan_correction,
        validate_plan,
    )
    from agent.planner import plan_query, planner_tool_calls_to_plan_calls
    from agent.semantic_layer.resolver import clarification_message, resolve_query
    from agent.semantic_layer.models import ResolvedQuery

    def _intent_from_resolved_query(rq: ResolvedQuery) -> dict[str, Any]:
        return {
            "resolved_metrics": [m.model_dump() for m in rq.resolved_metrics],
            "resolved_concepts": [c.model_dump() for c in rq.resolved_concepts],
            "resolved_heuristics": [h.model_dump() for h in rq.resolved_heuristics],
            "literals": [l.model_dump() for l in rq.literals],
            "unresolved_essential_terms": list(rq.unresolved_essential_terms),
            "unresolved_flavour_terms": list(rq.unresolved_flavour_terms),
            "needs_clarification": rq.needs_clarification,
            "comparison_mode": (
                "find_best"
                if any(c.key.startswith("buena_zona") for c in rq.resolved_concepts)
                else "explore"
            ),
            "recommendation_requested": any(
                c.key.startswith("buena_zona") for c in rq.resolved_concepts
            ),
            "depth_preference": "analytical",
        }

    model_name = model or SYNTHESIZER_MODEL_DEFAULT
    try:
        resolved_query = resolve_query(user_input, session_memory=conversation_state)
        resolved_intent_dict: dict[str, Any] | None = _intent_from_resolved_query(resolved_query)
        log_stage(
            "semantic_resolver",
            "resolved",
            metrics=[m.key for m in resolved_query.resolved_metrics],
            concepts=[c.key for c in resolved_query.resolved_concepts],
            heuristics=[h.key for h in resolved_query.resolved_heuristics],
            unresolved_essential_terms=resolved_query.unresolved_essential_terms,
            unresolved_flavour_terms=resolved_query.unresolved_flavour_terms,
        )
    except Exception:
        resolved_query = resolve_query("", session_memory=conversation_state)
        resolved_intent_dict = None
        log_stage("semantic_resolver", "failed", user_message=user_input)

    clarification = clarification_message(resolved_query)
    if clarification:
        pending = {
            "term": resolved_query.unresolved_essential_terms[0],
            "original_query": user_input,
        }
        log_stage(
            "semantic_resolver",
            "clarification_required",
            unresolved_essential_terms=resolved_query.unresolved_essential_terms,
            pending_clarification=pending,
        )
        return {
            "error": None,
            "conversational_text": clarification,
            "validated_plan": None,
            "execution_results": None,
            "max_tokens_final": 0,
            "validation_failed": False,
            "validation_errors": [],
            "had_validation_replan": False,
            "planner_response_id": None,
            "responses_synthesis": None,
            "resolved_intent": resolved_intent_dict,
            "pending_clarification": pending,
        }

    max_steps = compute_max_steps(resolved_intent_dict)
    correction_hint: str | None = None
    had_validation_replan = False

    all_execution: list[dict[str, Any]] = []
    validation_errors: list[str] = []
    last_validated: dict[str, Any] = {}
    planner_response_id: str | None = previous_planner_response_id
    completed_tool_path = False
    planned_call_count = 0

    for step in range(max_steps):
        previous_results = _build_results_summary_for_synth(all_execution[-8:])
        decision, response_id = plan_query(
            user_input,
            resolved_query=resolved_query,
            session_memory=conversation_state,
            live_schema_snapshot=f"{live_schema_context}\n\n=== STATIC ===\n{static_schema}",
            conversation_context=conversation_context,
            last_assistant_context=last_assistant_context,
            correction_hint=correction_hint,
            previous_results=previous_results,
            model=model_name,
            prompt_cache_key=prompt_cache_key,
            timeout_sec=timeout_sec,
        )
        planner_response_id = response_id or planner_response_id
        log_stage(
            "planner",
            "decision",
            step=step + 1,
            route=decision.route,
            tool_calls=[call.tool for call in decision.tool_calls],
            correction_hint=bool(correction_hint),
        )
        if decision.reasoning_summary and preamble_callback and decision.route == "data":
            _LOG.info("agent_reasoning_stream step=%s text=%s", step + 1, decision.reasoning_summary)
            log_stage(
                "planner",
                "reasoning_stream",
                step=step + 1,
                text=decision.reasoning_summary,
            )
            try:
                preamble_callback(decision.reasoning_summary)
            except Exception:
                pass

        if decision.route == "conversational":
            text = (decision.conversational_response or "").strip()
            if completed_tool_path:
                break
            return {
                "error": None,
                "conversational_text": text,
                "validated_plan": None,
                "execution_results": None,
                "max_tokens_final": 0,
                "validation_failed": False,
                "validation_errors": [],
                "had_validation_replan": had_validation_replan,
                "planner_response_id": planner_response_id,
                "responses_synthesis": None,
                "resolved_intent": resolved_intent_dict,
            }

        raw_plan_calls = planner_tool_calls_to_plan_calls(decision.tool_calls)
        if not raw_plan_calls:
            if completed_tool_path:
                break
            return {
                "error": None,
                "conversational_text": (decision.conversational_response or "").strip() or "No valid tool calls.",
                "validated_plan": None,
                "execution_results": None,
                "max_tokens_final": 0,
                "validation_failed": False,
                "validation_errors": [],
                "had_validation_replan": had_validation_replan,
                "planner_response_id": planner_response_id,
                "responses_synthesis": None,
                "resolved_intent": resolved_intent_dict,
            }

        plan: dict[str, Any] = {
            "tool_calls": raw_plan_calls,
            "reasoning": f"agent_loop_step_{step + 1}",
            "neighborhood_resolved": None,
            "combine_maps": False,
        }
        _infer_plan_neighborhood_resolved(plan)
        plan["combine_maps"] = _infer_combine_maps_from_tools(plan)

        validated = validate_plan(plan, live_schema_context)
        ve = list(validated.get("validation_errors") or [])
        validation_errors.extend(ve)
        log_stage(
            "validator",
            "validated",
            step=step + 1,
            tool_calls=[c.get("tool") for c in validated.get("tool_calls") or []],
            errors=ve,
        )
        if plan.get("tool_calls") and not (validated.get("tool_calls") or []):
            correction_hint = format_validation_plan_correction(ve, raw_plan_calls)
            had_validation_replan = True
            continue

        execution_results = execute_plan(validated, engine, user_input)
        log_stage(
            "executor",
            "executed",
            step=step + 1,
            results=[
                {
                    "tool": r.get("tool"),
                    "success": r.get("success"),
                    "row_count": r.get("row_count"),
                    "error": r.get("error"),
                }
                for r in execution_results
            ],
        )
        planned_call_count += len(raw_plan_calls)
        all_execution.extend(execution_results)
        last_validated = validated
        completed_tool_path = True
        # Let real execution results drive self-correction on the next loop.
        if step < max_steps - 1 and any(not r.get("success") for r in execution_results):
            correction_hint = (
                "Previous tool execution returned errors. Read those errors and correction hints, "
                "then emit corrected tool_calls."
            )
            continue
        correction_hint = None

        # Most Rooster questions are answered by one validated parallel batch. Continue only
        # for recommendation/find-best cases where the model may need a follow-up query.
        if max_steps <= 1 or step >= max_steps - 1:
            break
        if not (
            resolved_intent_dict
            and (
                resolved_intent_dict.get("comparison_mode") == "find_best"
                or resolved_intent_dict.get("recommendation_requested") is True
            )
        ):
            break

    if not completed_tool_path:
        return {
            "error": None,
            "conversational_text": "Lo siento, no pude procesar tu mensaje. ¿Puedes reformular?",
            "validated_plan": None,
            "execution_results": None,
            "max_tokens_final": 0,
            "validation_failed": False,
            "validation_errors": validation_errors,
            "had_validation_replan": had_validation_replan,
            "planner_response_id": planner_response_id,
            "responses_synthesis": None,
            "resolved_intent": resolved_intent_dict,
        }

    last_validated = dict(last_validated or {})
    last_validated["agent_loop_steps"] = min(max_steps, max(1, planned_call_count))
    return {
        "error": None,
        "conversational_text": None,
        "validated_plan": last_validated,
        "execution_results": all_execution,
        "max_tokens_final": 0,
        "validation_failed": False,
        "validation_errors": validation_errors,
        "had_validation_replan": had_validation_replan,
        "planner_response_id": planner_response_id,
        "responses_synthesis": None,
        "resolved_intent": resolved_intent_dict,
    }
