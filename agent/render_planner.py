"""Typed RenderPlan composer for Rooster v2."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from agent.agent_pipeline import build_render_stack


BlockType = Literal[
    "kpi_strip",
    "narrative",
    "comparison_table",
    "listing_table",
    "map",
    "chart",
    "score_breakdown",
    "knowledge_context",
    "legacy",
]


class Block(BaseModel):
    type: BlockType
    body_id: str | None = None
    role: str | None = None
    title: str | None = None
    source_tool: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


class KPIStripBlock(Block):
    type: Literal["kpi_strip"] = "kpi_strip"


class NarrativeBlock(Block):
    type: Literal["narrative"] = "narrative"
    body_id: str
    role: str = "analysis"


class ComparisonTableBlock(Block):
    type: Literal["comparison_table"] = "comparison_table"


class ListingTableBlock(Block):
    type: Literal["listing_table"] = "listing_table"


class MapBlock(Block):
    type: Literal["map"] = "map"


class ChartBlock(Block):
    type: Literal["chart"] = "chart"


class ScoreBreakdownBlock(Block):
    type: Literal["score_breakdown"] = "score_breakdown"


class KnowledgeContextBlock(Block):
    type: Literal["knowledge_context"] = "knowledge_context"


class RenderPlan(BaseModel):
    blocks: list[Block] = Field(default_factory=list)
    follow_ups: list[str] = Field(default_factory=list)


def _block_for_legacy_stack_item(item: dict[str, Any], idx: int) -> Block:
    intent = str(item.get("intent") or "search")
    rows = item.get("rows") or []
    meta = dict(item.get("meta") or {})
    tool = str(meta.get("tool") or "")
    payload = {
        "legacy_intent": intent,
        "rows": rows,
        "meta": meta,
    }
    if intent in {"geo", "transit_map", "tourism_map", "combined_map", "neighborhood_highlight"}:
        return MapBlock(source_tool=tool, payload=payload, title="Mapa")
    if intent in {"ranking", "profile_scatter", "chart", "trend"}:
        return ChartBlock(source_tool=tool, payload=payload, title="Gráfico")
    if intent == "overview":
        return KPIStripBlock(source_tool=tool, payload=payload, title="Indicadores")
    if tool == "compare_neighborhoods":
        return ComparisonTableBlock(source_tool=tool, payload=payload, title="Comparación")
    if tool == "query_listings" or intent in {"search", "no_coords"}:
        return ListingTableBlock(source_tool=tool, payload=payload, title="Anuncios")
    if tool == "query_neighborhood_context":
        return KnowledgeContextBlock(source_tool=tool, payload=payload, title="Contexto")
    return Block(type="legacy", source_tool=tool, payload=payload, title=f"Bloque {idx + 1}")


def lint_render_plan(plan: RenderPlan) -> RenderPlan:
    """Conservative Python lint: cap maps/KPI strips and drop contradictory duplicates."""
    blocks: list[Block] = []
    map_count = 0
    kpi_count = 0
    seen_keys: set[tuple[str, str, str]] = set()
    for block in plan.blocks:
        if block.type == "map":
            map_count += 1
            if map_count > 2:
                continue
        if block.type == "kpi_strip":
            kpi_count += 1
            if kpi_count > 1:
                continue
        legacy_intent = str(block.payload.get("legacy_intent") or "")
        source_tool = str(block.source_tool or "")
        key = (block.type, source_tool, legacy_intent)
        if key in seen_keys and block.type != "narrative":
            continue
        seen_keys.add(key)
        blocks.append(block)
    return RenderPlan(blocks=blocks, follow_ups=plan.follow_ups[:3])


def compose_render_plan(
    validated_plan: dict[str, Any],
    execution_results: list[dict[str, Any]],
    geo_key: int,
    *,
    resolved_intent: dict[str, Any] | None = None,
    memory: dict[str, Any] | None = None,
) -> RenderPlan:
    """Compose a deterministic RenderPlan over existing v1 renderer primitives."""
    del memory
    stack = build_render_stack(validated_plan, execution_results, geo_key)
    blocks: list[Block] = []
    has_data = any((res.get("row_count") or 0) > 0 for res in execution_results or [])
    if has_data:
        blocks.append(
            NarrativeBlock(
                body_id="summary",
                role="analysis",
                title="Resumen",
                payload={"resolved_intent": resolved_intent or {}},
            )
        )
    blocks.extend(_block_for_legacy_stack_item(item, i) for i, item in enumerate(stack))

    tools = {str(r.get("tool") or "") for r in execution_results or []}
    follow_ups: list[str] = []
    if "query_listings" in tools:
        follow_ups.append("Ver estos anuncios en mapa")
    if "query_neighborhood_profile" in tools or "compare_neighborhoods" in tools:
        follow_ups.append("Comparar con barrios similares")
    if "query_neighborhood_density_chart" not in tools:
        follow_ups.append("Ver presión turística")
    return lint_render_plan(RenderPlan(blocks=blocks, follow_ups=follow_ups))
