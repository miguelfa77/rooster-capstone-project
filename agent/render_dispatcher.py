"""Composable RenderPlan dispatcher over Rooster's existing renderer primitives."""

from __future__ import annotations

from typing import Any

from agent.render_planner import RenderPlan, compose_render_plan


def build_blocks_from_results(
    validated_plan: dict[str, Any],
    execution_results: list[dict[str, Any]],
    geo_key: int,
    resolved_intent: dict[str, Any] | None = None,
    memory: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], RenderPlan]:
    plan = compose_render_plan(
        validated_plan,
        execution_results,
        geo_key,
        resolved_intent=resolved_intent,
        memory=memory,
    )
    return render_plan_to_legacy_stack(plan), plan


def render_plan_to_legacy_stack(plan: RenderPlan) -> list[dict[str, Any]]:
    """
    Convert typed blocks to the legacy stack shape consumed by existing Streamlit dispatch.
    Narrative blocks are handled by synthesis/prose, so they do not emit visual blocks.
    """
    stack: list[dict[str, Any]] = []
    for block in plan.blocks:
        if block.type == "narrative":
            continue
        payload = block.payload or {}
        legacy_intent = payload.get("legacy_intent")
        if legacy_intent:
            if block.type == "comparison_table":
                legacy_intent = "comparison_table"
            elif block.type == "kpi_strip":
                legacy_intent = "kpi_strip"
            elif block.type == "score_breakdown":
                legacy_intent = "score_breakdown"
            elif block.type == "chart" and block.source_tool == "query_neighborhood_density_chart":
                legacy_intent = "density_chart"
            stack.append(
                {
                    "intent": legacy_intent,
                    "rows": payload.get("rows") or [],
                    "meta": dict(payload.get("meta") or {}),
                    "render_block_type": block.type,
                    "render_block_title": block.title,
                }
            )
            continue
        if block.type == "comparison_table":
            stack.append(
                {
                    "intent": "search",
                    "rows": payload.get("rows") or [],
                    "meta": dict(payload.get("meta") or {}),
                }
            )
        elif block.type == "listing_table":
            stack.append(
                {
                    "intent": "search",
                    "rows": payload.get("rows") or [],
                    "meta": dict(payload.get("meta") or {}),
                }
            )
    return stack
