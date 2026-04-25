"""Bounded v2 Responses tool loop for Rooster."""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Any

MAX_STEPS_DEFAULT = int(os.environ.get("ROOSTER_MAX_AGENT_STEPS", "3"))


def compute_max_steps(resolved_intent: dict[str, Any] | None) -> int:
    if not resolved_intent:
        return min(MAX_STEPS_DEFAULT, 3)
    if resolved_intent.get("comparison_mode") == "find_best" or resolved_intent.get(
        "recommendation_requested"
    ):
        return min(5, max(MAX_STEPS_DEFAULT, 5))
    return min(MAX_STEPS_DEFAULT, 3)


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
    Phase 2/3 loop: planner -> validate -> execute -> feed tool outputs back while
    budget remains. Rendering is composed later by RenderPlan.
    """
    from agent.agent_pipeline import (
        _build_results_summary_for_synth,
        _infer_combine_maps_from_tools,
        _infer_plan_neighborhood_resolved,
        _openai_fc_first_completion,
        build_openai_first_turn_messages,
        execute_plan,
        format_output_completeness_correction,
        format_validation_plan_correction,
        openai_tool_calls_to_plan_calls,
        validate_output_completeness,
        validate_plan,
    )
    from agent.intent_resolver import resolve_intent
    from agent.responses_api import (
        chat_tools_to_responses_tools,
        create_response_with_tools,
        extract_response_text,
        get_openai_client,
        output_items_to_tool_calls_compat,
        tool_json_payloads_to_responses_input,
    )
    from agent.openai_tools import get_rooster_openai_tools

    model_name = model or os.getenv("SYNTHESISER_MODEL", "gpt-5.5")
    combined_schema = f"{live_schema_context}\n\n=== STATIC ===\n{static_schema}"
    try:
        ri = resolve_intent(
            user_input,
            conversation_state,
            combined_schema,
            timeout_sec=min(float(timeout_sec), 25.0),
            prompt_cache_key=prompt_cache_key,
        )
        resolved_intent_dict: dict[str, Any] | None = ri.model_dump()
    except Exception:
        resolved_intent_dict = None

    max_steps = compute_max_steps(resolved_intent_dict)
    correction_hint: str | None = None
    had_output_correction = False
    had_validation_replan = False

    all_tcalls: list[Any] = []
    all_execution: list[dict[str, Any]] = []
    validation_errors: list[str] = []
    last_validated: dict[str, Any] = {}
    planner_response_id: str | None = previous_planner_response_id
    completed_tool_path = False
    last_response: Any = None

    client = get_openai_client(max(timeout_sec, 35.0))
    tools = chat_tools_to_responses_tools(get_rooster_openai_tools())

    for step in range(max_steps):
        if step == 0 or correction_hint:
            first_messages = build_openai_first_turn_messages(
                user_input,
                conversation_state,
                conversation_context,
                live_schema_context,
                static_schema,
                correction_hint=correction_hint,
                last_assistant_context=last_assistant_context,
                resolved_intent=resolved_intent_dict,
            )

            def _run_first() -> Any:
                return _openai_fc_first_completion(
                    first_messages,
                    model_name,
                    timeout_sec,
                    prompt_cache_key=prompt_cache_key,
                    previous_response_id=planner_response_id,
                )

            try:
                with ThreadPoolExecutor(max_workers=1) as ex:
                    response = ex.submit(_run_first).result(timeout=timeout_sec)
            except FuturesTimeoutError:
                return {
                    "error": "timeout",
                    "had_validation_replan": had_validation_replan,
                    "resolved_intent": resolved_intent_dict,
                }
        else:
            followup_user = (
                "Ya tienes estos resultados. Si falta una consulta necesaria para responder, "
                "llama otra herramienta. Si no falta nada, responde solo DONE."
            )
            pairs: list[tuple[str, str]] = []
            for tc in all_tcalls[-8:]:
                er = next(
                    (r for r in all_execution if r.get("tool_call_id") == getattr(tc, "id", None)),
                    None,
                )
                summ = _build_results_summary_for_synth([er]) if er else []
                payload = summ[0] if summ else {"error": "sin resultado"}
                pairs.append((str(getattr(tc, "id", "")), json.dumps(payload, default=str)))
            response = create_response_with_tools(
                client,
                model=model_name,
                instructions="Continúa el bucle de herramientas de Rooster solo si falta información.",
                user_input=tool_json_payloads_to_responses_input(pairs, followup_user),
                tools=tools,
                previous_response_id=planner_response_id,
                prompt_cache_key=prompt_cache_key,
                max_output_tokens=700,
                parallel_tool_calls=True,
                reasoning_effort="medium",
            )

        planner_response_id = getattr(response, "id", None) or planner_response_id
        last_response = response
        tcalls = output_items_to_tool_calls_compat(response)
        rtext = extract_response_text(response)
        if tcalls and rtext and preamble_callback:
            try:
                preamble_callback(rtext.strip())
            except Exception:
                pass

        if not tcalls:
            text = (rtext or "").strip()
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
                "had_output_correction": False,
                "had_validation_replan": had_validation_replan,
                "planner_response_id": planner_response_id,
                "responses_synthesis": None,
                "resolved_intent": resolved_intent_dict,
            }

        raw_plan_calls = openai_tool_calls_to_plan_calls(tcalls)
        if not raw_plan_calls:
            if completed_tool_path:
                break
            return {
                "error": None,
                "conversational_text": (rtext or "").strip() or "No valid tool calls.",
                "validated_plan": None,
                "execution_results": None,
                "max_tokens_final": 0,
                "validation_failed": False,
                "validation_errors": [],
                "had_output_correction": False,
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
        if plan.get("tool_calls") and not (validated.get("tool_calls") or []):
            correction_hint = format_validation_plan_correction(ve, raw_plan_calls)
            had_validation_replan = True
            continue

        execution_results = execute_plan(validated, engine, user_input)
        issues = validate_output_completeness(user_input, execution_results)
        if issues:
            correction_hint = format_output_completeness_correction(issues, user_input)
            had_output_correction = True
            continue

        all_tcalls.extend(tcalls)
        all_execution.extend(execution_results)
        last_validated = validated
        completed_tool_path = True
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
            "had_output_correction": had_output_correction,
            "had_validation_replan": had_validation_replan,
            "planner_response_id": planner_response_id,
            "responses_synthesis": None,
            "resolved_intent": resolved_intent_dict,
        }

    last_validated = dict(last_validated or {})
    last_validated["agent_loop_steps"] = min(max_steps, max(1, len(all_tcalls)))
    return {
        "error": None,
        "conversational_text": None,
        "validated_plan": last_validated,
        "execution_results": all_execution,
        "max_tokens_final": 0,
        "validation_failed": False,
        "validation_errors": validation_errors,
        "had_output_correction": had_output_correction,
        "had_validation_replan": had_validation_replan,
        "planner_response_id": planner_response_id or getattr(last_response, "id", None),
        "responses_synthesis": None,
        "resolved_intent": resolved_intent_dict,
    }
