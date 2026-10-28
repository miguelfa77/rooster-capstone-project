"""One chat turn, independent of any UI.

`run_turn` runs the agent loop, synthesis, code-primitive correction and memory
update for a single user message, reporting progress through `emit`. Streamlit
and the HTTP API are both clients of this module.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field, replace
from typing import Annotated, Any, Callable, Literal, Union

from pydantic import BaseModel, Field

from agent import ui_es as UI
from agent.agent_loop import run_agent_loop_pipeline
from agent.agent_pipeline import (
    extract_neighborhood_names_from_schema,
    format_last_assistant_for_planner,
    strip_follow_ups_suffix,
)
from agent.conversation_context import (
    build_conversation_context,
    format_conversation_for_memo,
    is_summary_request,
)
from agent.llm_sql import summarize_conversation_memo
from agent.sandbox_client import call_sandbox
from agent.session_memory import coalesce_session_memory, update_session_memory_from_turn
from agent.stage_logging import log_stage
from agent.synthesizer import (
    CodePrimitive,
    SynthesizedResponse,
    TextPrimitive,
    _fallback_response,
    _sanitize_response,
    stream_synthesize_response,
    synthesize_response,
    synthesized_text,
)

_LOG = logging.getLogger("rooster.turn")


# ── Events ────────────────────────────────────────────────────────────────


class ThoughtEvent(BaseModel):
    type: Literal["thought"] = "thought"
    text: str


class TextDeltaEvent(BaseModel):
    type: Literal["text_delta"] = "text_delta"
    text: str


class PrimitiveEvent(BaseModel):
    type: Literal["primitive"] = "primitive"
    primitive: dict[str, Any]


class FollowUpsEvent(BaseModel):
    type: Literal["follow_ups"] = "follow_ups"
    items: list[str]


class ErrorEvent(BaseModel):
    type: Literal["error"] = "error"
    reason: str
    suggestion: str


class DoneEvent(BaseModel):
    type: Literal["done"] = "done"


TurnEvent = Annotated[
    Union[ThoughtEvent, TextDeltaEvent, PrimitiveEvent, FollowUpsEvent, ErrorEvent, DoneEvent],
    Field(discriminator="type"),
]

Emit = Callable[[Any], None]


# ── State and context ─────────────────────────────────────────────────────


@dataclass
class TurnState:
    messages: list[dict] = field(default_factory=list)
    session_memory: dict = field(default_factory=dict)
    previous_execution_results: list[dict] = field(default_factory=list)
    prompt_cache_key: str = ""

    @classmethod
    def new(cls) -> "TurnState":
        return cls(prompt_cache_key=f"rooster_{uuid.uuid4().hex[:16]}")


@dataclass
class TurnContext:
    engine: Any
    live_schema: str
    static_schema: str
    model: str
    timeout_sec: float


# ── Turn ──────────────────────────────────────────────────────────────────


def _base_message(**fields: Any) -> dict:
    msg = {
        "role": "assistant",
        "intent": "conversational",
        "summary": "",
        "render_stack": [],
        "validation_errors": [],
        "rows": None,
        "sql": None,
        "error": None,
        "empty": False,
        "followups": [],
        "follow_ups": list(UI.FOLLOW_UP_DEFAULTS),
        "confidence": "high",
        "interpretation": "",
        "reasoning_focus": "",
        "caveat": "",
    }
    msg.update(fields)
    return msg


def _fallback_message(reason: str, suggestion: str, emit: Emit, **fields: Any) -> dict:
    emit(ErrorEvent(reason=reason, suggestion=suggestion))
    return _base_message(graceful_fallback={"reason": reason, "suggestion": suggestion}, **fields)


def _update_memory(state: TurnState, user_input: str, response_text: str, tools_used: list[str]) -> dict:
    """Return updated session memory; on any failure keep the previous memory."""
    try:
        mem0 = coalesce_session_memory(state.session_memory)
        mem1 = update_session_memory_from_turn(
            mem0,
            user_input,
            response_text or "",
            tools_used=tools_used,
            timeout_sec=20.0,
            prompt_cache_key=state.prompt_cache_key,
        )
        log_stage(
            "memory",
            "session_memory_updated",
            tools_used=tools_used,
            neighborhoods=mem1.conversation_state.neighborhoods_in_focus,
            priorities=mem1.user_profile.inferred_priorities,
        )
        return mem1.model_dump()
    except Exception:
        _LOG.warning("Session memory update failed", exc_info=True)
        return state.session_memory


def _try_correct_code_primitives(
    synthesized: SynthesizedResponse,
    *,
    execution_results: list[dict],
    user_message: str,
    resolved_intent: dict | None,
    session_memory: dict | None,
    ctx: TurnContext,
    prompt_cache_key: str,
) -> SynthesizedResponse:
    """Pre-check code primitives against the sandbox; re-synthesize once if any fail."""
    rows: list[dict] = []
    for r in execution_results or []:
        if r.get("success") and r.get("rows"):
            rows = r["rows"]
            break
    if not rows:
        return synthesized

    for primitive in synthesized.primitives:
        if not isinstance(primitive, CodePrimitive):
            continue
        result = call_sandbox(primitive.code, rows)
        if not result.get("success"):
            error = result.get("error") or "Unknown sandbox error"
            columns = sorted({k for row in rows[:3] if isinstance(row, dict) for k in row})
            _LOG.warning("Code primitive failed pre-check, re-synthesizing: %s", error)
            correction_block = (
                f"Your generated code failed with this error when executed:\n\n"
                f"{error}\n\n"
                f"Available columns in df: {columns}\n\n"
                f"Rewrite only the code primitive to fix this error."
            )
            return synthesize_response(
                user_message,
                resolved_intent=resolved_intent,
                agent_results=execution_results,
                session_memory=session_memory,
                model=ctx.model,
                timeout_sec=ctx.timeout_sec,
                prompt_cache_key=prompt_cache_key,
                correction_block=correction_block,
            )
    return synthesized


def _memo_turn(user_input: str, state: TurnState, ctx: TurnContext, emit: Emit) -> dict:
    history = state.messages + [{"role": "user", "content": user_input}]
    try:
        memo_text = summarize_conversation_memo(
            format_conversation_for_memo(history),
            ctx.model,
            {},
            timeout_sec=min(float(ctx.timeout_sec), 60.0),
        )
    except Exception as e:
        memo_text = UI.MEMO_ERR.format(err=e)
    emit(TextDeltaEvent(text=memo_text))
    return _base_message(is_memo=True, summary=memo_text, intent="memo")


def _agent_turn(user_input: str, state: TurnState, ctx: TurnContext, emit: Emit) -> tuple[dict, TurnState]:
    """Run the agent for one message. Returns the assistant message and the updated state fields."""
    memory_context = state.session_memory or {}
    fc = run_agent_loop_pipeline(
        user_input,
        memory_context,
        ctx.live_schema,
        ctx.static_schema,
        build_conversation_context(state.messages),
        ctx.model,
        float(ctx.timeout_sec),
        ctx.engine,
        last_assistant_context=format_last_assistant_for_planner(state.messages),
        prompt_cache_key=state.prompt_cache_key,
        previous_planner_response_id=None,
        preamble_callback=lambda text: emit(ThoughtEvent(text=str(text or "").strip()))
        if str(text or "").strip()
        else None,
        previous_execution_results=state.previous_execution_results,
    )

    if fc.get("error") == "timeout":
        msg = _fallback_message(UI.CHAT_TIMEOUT_REASON, UI.CHAT_TIMEOUT_SUGGESTION, emit, confidence="low")
        return msg, state

    if fc.get("validation_failed"):
        top = extract_neighborhood_names_from_schema(ctx.live_schema)[:5]
        names_sample = ", ".join(top) if top else "…"
        ve = fc.get("validation_errors") or []
        reason = (
            "No pude encontrar ese barrio en mis datos. "
            f"Algunos barrios disponibles incluyen: {names_sample}…"
        )
        if ve:
            reason += " " + " ".join(str(x) for x in ve[:3])
        msg = _fallback_message(
            reason, UI.CHAT_VALIDATION_FAILED_SUGGESTION, emit, agent_turn=True, validation_errors=ve
        )
        return msg, state

    if fc.get("conversational_text") is not None:
        prose, follow_ups = strip_follow_ups_suffix(fc["conversational_text"] or "")
        follow_ups = follow_ups or list(UI.FOLLOW_UP_DEFAULTS)
        emit(TextDeltaEvent(text=prose))
        emit(FollowUpsEvent(items=follow_ups))
        memory = _update_memory(state, user_input, prose, [])
        return _base_message(summary=prose, follow_ups=follow_ups), replace(state, session_memory=memory)

    if fc.get("execution_results") is None:
        _LOG.warning("Agent loop returned an unexpected result: %s", fc)
        msg = _fallback_message(
            "Lo siento, no pude procesar tu mensaje.", UI.CHAT_UNEXPECTED_SUGGESTION, emit, confidence="low"
        )
        return msg, state

    validated = fc.get("validated_plan") or {}
    execution_results = fc["execution_results"] or []
    validation_errors = list(validated.get("validation_errors") or [])

    synthesized: SynthesizedResponse | None = None
    for item in stream_synthesize_response(
        user_input,
        resolved_intent=fc.get("resolved_intent"),
        agent_results=execution_results,
        session_memory=memory_context,
        model=ctx.model,
        timeout_sec=float(ctx.timeout_sec),
        prompt_cache_key=state.prompt_cache_key,
    ):
        if isinstance(item, str):
            emit(TextDeltaEvent(text=item))
        elif isinstance(item, SynthesizedResponse):
            synthesized = item
    if synthesized is None:
        synthesized = _sanitize_response(_fallback_response(execution_results))

    synthesized = _try_correct_code_primitives(
        synthesized,
        execution_results=execution_results,
        user_message=user_input,
        resolved_intent=fc.get("resolved_intent"),
        session_memory=memory_context,
        ctx=ctx,
        prompt_cache_key=state.prompt_cache_key,
    )
    for primitive in synthesized.primitives:
        if not isinstance(primitive, TextPrimitive):
            emit(PrimitiveEvent(primitive=primitive.model_dump()))

    response_text = synthesized_text(synthesized)
    follow_ups = synthesized.follow_ups or list(UI.FOLLOW_UP_DEFAULTS)
    emit(FollowUpsEvent(items=follow_ups))

    tools_used = [
        c.get("tool") for c in validated.get("tool_calls") or [] if isinstance(c, dict) and c.get("tool")
    ]
    memory = _update_memory(state, user_input, response_text, tools_used)

    payload = synthesized.model_dump()
    first_visual = next(
        (p for p in payload.get("primitives", []) if p.get("kind") in {"code", "table", "kpi"}),
        {},
    )
    msg = _base_message(
        agent_turn=True,
        summary=response_text or "",
        primitive_response=payload,
        execution_results=execution_results,
        validation_errors=validation_errors,
        intent=first_visual.get("kind") or "conversational",
        follow_ups=follow_ups,
    )
    return msg, replace(state, session_memory=memory, previous_execution_results=execution_results)


def run_turn(user_input: str, state: TurnState, ctx: TurnContext, emit: Emit) -> TurnState:
    """Run one turn. Never raises; always emits DoneEvent last."""
    user_msg = {"role": "user", "content": user_input}
    try:
        if is_summary_request(user_input):
            msg, new_state = _memo_turn(user_input, state, ctx, emit), state
        else:
            msg, new_state = _agent_turn(user_input, state, ctx, emit)
    except Exception as e:
        _LOG.exception("Chat turn failed: %s", e)
        msg = _fallback_message(str(e), UI.CHAT_EXCEPTION_SUGGESTION, emit, intent="search")
        new_state = state
    emit(DoneEvent())
    return replace(new_state, messages=state.messages + [user_msg, msg])
