"""
Dispatch composable render blocks. Today this delegates to ``build_render_stack`` + ``dispatch`` in the UI.
"""

from __future__ import annotations

from typing import Any

from agent.agent_pipeline import build_render_stack
from agent.render_planner import RenderPlan, compose_render_plan


def build_blocks_from_results(
    validated_plan: dict[str, Any],
    execution_results: list[dict[str, Any]],
    geo_key: int,
    resolved_intent: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], RenderPlan]:
    plan = compose_render_plan(
        validated_plan,
        execution_results,
        geo_key,
        resolved_intent=resolved_intent,
    )
    return build_render_stack(validated_plan, execution_results, geo_key), plan
