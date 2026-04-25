"""
Structured session memory (v2) + post-turn merge helpers.
"""

from __future__ import annotations

import json
import os
from typing import Any

from pydantic import BaseModel, Field

from agent.responses_api import get_openai_client

MEMORY_MODEL = os.getenv("ROOSTER_MEMORY_MODEL", "gpt-4o-mini")


class UserProfileModel(BaseModel):
    inferred_operation: str | None = None
    inferred_price_band: dict[str, Any] | None = None
    inferred_priorities: dict[str, float] = Field(default_factory=dict)
    explicit_constraints: list[str] = Field(default_factory=list)


class ConversationStateModel(BaseModel):
    neighborhoods_in_focus: list[str] = Field(default_factory=list)
    last_comparison: dict[str, Any] | None = None
    pending_threads: list[str] = Field(default_factory=list)
    turn: int = 0
    stage: str = "orienting"


class ShownSoFarModel(BaseModel):
    tools_called: list[str] = Field(default_factory=list)
    neighborhoods_shown: list[str] = Field(default_factory=list)
    last_render_plan_summary: str = ""


class SessionMemoryV2(BaseModel):
    user_profile: UserProfileModel = Field(default_factory=UserProfileModel)
    conversation_state: ConversationStateModel = Field(default_factory=ConversationStateModel)
    shown_so_far: ShownSoFarModel = Field(default_factory=ShownSoFarModel)


class MemoryTurnDelta(BaseModel):
    user_profile: UserProfileModel | None = None
    conversation_state: ConversationStateModel | None = None
    shown_so_far: ShownSoFarModel | None = None


MEMORY_UPDATE_INSTRUCTIONS = """You merge new signals into Rooster session memory. Output JSON matching the schema.
Spanish real estate (Valencia). Be conservative: only set fields you can justify from the user message and assistant summary.
inferred_operation: venta|alquiler|either|null
inferred_priorities keys may include yield, transit, tourism_avoidance, price_sensitivity in 0..1
stage: orienting|evaluating|deciding
Turn counter and thread lists: short Spanish phrases for pending threads.
"""


def _empty_profile() -> dict[str, Any]:
    return {
        "inferred_operation": None,
        "inferred_price_band": None,
        "inferred_priorities": {},
        "explicit_constraints": [],
    }


def coalesce_session_memory(st: Any) -> SessionMemoryV2:
    if isinstance(st, SessionMemoryV2):
        return st
    if isinstance(st, dict) and st and "user_profile" in st:
        try:
            return SessionMemoryV2.model_validate(st)
        except Exception:
            pass
    return SessionMemoryV2()


def merge_session_memory(
    current: SessionMemoryV2,
    delta: MemoryTurnDelta,
) -> SessionMemoryV2:
    d = current.model_dump()
    if delta.user_profile:
        up = d["user_profile"]
        mp = delta.user_profile.model_dump(exclude_unset=True)
        for k, v in mp.items():
            if v is not None and v != "" and v != []:
                up[k] = v
    if delta.conversation_state:
        cs = d["conversation_state"]
        mcs = delta.conversation_state.model_dump(exclude_unset=True)
        for k, v in mcs.items():
            if v is not None and v != "" and v != []:
                cs[k] = v
    if delta.shown_so_far:
        sh = d["shown_so_far"]
        msh = delta.shown_so_far.model_dump(exclude_unset=True)
        for k, v in msh.items():
            if v is not None and v != "" and v != []:
                sh[k] = v
    return SessionMemoryV2.model_validate(d)


def update_session_memory_from_turn(
    current: SessionMemoryV2,
    user_message: str,
    assistant_summary: str,
    *,
    tools_used: list[str],
    timeout_sec: float = 12.0,
    prompt_cache_key: str | None = None,
) -> SessionMemoryV2:
    """Structured-outputs call to produce a diff and merge."""
    if not (os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_KEY")):
        return current

    client = get_openai_client(timeout_sec)
    text_in = f"""User message:
{user_message}

Assistant summary (prose, may be truncated):
{assistant_summary[:6000]}

Tools called this turn (names): {json.dumps(tools_used)}
"""
    try:
        parse_kw: dict[str, Any] = {
            "model": MEMORY_MODEL,
            "instructions": MEMORY_UPDATE_INSTRUCTIONS,
            "input": text_in,
            "text_format": MemoryTurnDelta,
            "max_output_tokens": 1200,
            "temperature": 0,
        }
        if prompt_cache_key:
            parse_kw["prompt_cache_key"] = prompt_cache_key
        parsed = client.responses.parse(**parse_kw)
        delta = parsed.output_parsed
        if not isinstance(delta, MemoryTurnDelta):
            return current
        return merge_session_memory(current, delta)
    except Exception:
        return current


def sync_flat_conversation_state_v1(
    flat: dict[str, Any], mem: SessionMemoryV2
) -> dict[str, Any]:
    """Dual-write: keep v1 ``conversation_state`` keys updated from v2 for legacy code."""
    out = dict(flat)
    cs = mem.conversation_state
    up = mem.user_profile
    if cs.neighborhoods_in_focus:
        for n in cs.neighborhoods_in_focus:
            if n and n not in (out.get("neighborhoods_discussed") or []):
                out.setdefault("neighborhoods_discussed", [])
                if n not in out["neighborhoods_discussed"]:
                    out["neighborhoods_discussed"].append(n)
    if up.inferred_operation in ("venta", "alquiler", "either"):
        out["operation_focus"] = up.inferred_operation
    if cs.turn:
        out["turns"] = max(int(out.get("turns") or 0), cs.turn)
    if cs.stage in ("orienting", "evaluating", "deciding"):
        out["stage"] = cs.stage
    return out
