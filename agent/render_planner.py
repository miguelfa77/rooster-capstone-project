"""
Typed render plan (v2) — extends legacy ``build_render_stack`` with a JSON-serializable plan.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field

from agent.agent_pipeline import build_render_stack


class RenderBlock(BaseModel):
    type: str = Field(
        description="KPIStrip|Narrative|ComparisonTable|Map|Chart|ScoreBreakdown|…"
    )
    payload: dict[str, Any] = Field(default_factory=dict)


class RenderPlan(BaseModel):
    blocks: list[RenderBlock] = Field(default_factory=list)
    follow_ups: list[str] = Field(default_factory=list)


def compose_render_plan(
    validated_plan: dict[str, Any],
    execution_results: list[dict[str, Any]],
    geo_key: int,
    *,
    resolved_intent: dict[str, Any] | None = None,
) -> RenderPlan:
    """
    For now, derive blocks from the existing render stack; ``resolved_intent`` is reserved
    for adaptive composition.
    """
    del resolved_intent
    stack = build_render_stack(validated_plan, execution_results, geo_key)
    blocks = [
        RenderBlock(
            type="legacy_stack",
            payload={"intent": b.get("intent"), "rows_hint": len(b.get("rows") or [])},
        )
        for b in stack
    ]
    return RenderPlan(
        blocks=blocks,
        follow_ups=[],
    )


def render_plan_to_json(rp: RenderPlan) -> str:
    return json.dumps(rp.model_dump(), ensure_ascii=True, default=str)
