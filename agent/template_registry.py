"""Closed visual template registry for question-driven rendering."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

TemplateName = Literal[
    "ranking_bars",
    "comparison_matrix",
    "choropleth_focus",
    "point_map",
    "kpi_strip",
    "single_kpi",
    "score_breakdown",
    "histogram",
    "delta_table",
    "trend_line",
    "listing_cards",
    "listing_table",
    "scatter_2d",
    "none",
]


@dataclass(frozen=True)
class TemplateSpec:
    name: TemplateName
    purpose: str
    renderer_intent: str | None
    min_items: int = 0
    max_items: int | None = None
    requires_coordinates: bool = False
    requires_metric: bool = False
    required_time_points: int = 0
    default_meta: dict[str, Any] = field(default_factory=dict)


TEMPLATE_REGISTRY: dict[TemplateName, TemplateSpec] = {
    "ranking_bars": TemplateSpec(
        "ranking_bars",
        "Ordered magnitude comparison across 4-12 items; use for ranking claims with non-trivial spread.",
        "ranking",
        min_items=4,
        max_items=12,
        requires_metric=True,
    ),
    "comparison_matrix": TemplateSpec(
        "comparison_matrix",
        "Side-by-side metrics across 2-5 named items; use for multi-dimensional comparison claims.",
        "comparison_table",
        min_items=2,
        max_items=5,
    ),
    "choropleth_focus": TemplateSpec(
        "choropleth_focus",
        "Spatial pattern of a metric across barrios, with named items highlighted.",
        "neighborhood_highlight",
        min_items=2,
        requires_metric=True,
    ),
    "point_map": TemplateSpec(
        "point_map",
        "Where individual listings or point assets are located.",
        "geo",
        min_items=2,
        requires_coordinates=True,
    ),
    "kpi_strip": TemplateSpec(
        "kpi_strip",
        "Two to four headline numbers for lookup claims with multiple key figures.",
        "kpi_strip",
        min_items=1,
        max_items=4,
    ),
    "single_kpi": TemplateSpec(
        "single_kpi",
        "One large number with optional delta for a single-value lookup.",
        "single_kpi",
        min_items=1,
        max_items=1,
        requires_metric=True,
    ),
    "score_breakdown": TemplateSpec(
        "score_breakdown",
        "Decompose investment_score into yield, transport, and tourism components.",
        "score_breakdown",
        min_items=1,
        max_items=1,
    ),
    "histogram": TemplateSpec(
        "histogram",
        "Distribution of a metric across barrios or listings; use for typicality and spread claims.",
        "histogram",
        min_items=5,
        requires_metric=True,
    ),
    "delta_table": TemplateSpec(
        "delta_table",
        "Two snapshots side-by-side with percentage change.",
        "delta_table",
        min_items=2,
    ),
    "trend_line": TemplateSpec(
        "trend_line",
        "Time series of a metric; use when there are at least three time points.",
        "trend_line",
        min_items=3,
        requires_metric=True,
        required_time_points=3,
    ),
    "listing_cards": TemplateSpec(
        "listing_cards",
        "One to three highlighted listings with key fields.",
        "listing_cards",
        min_items=1,
        max_items=3,
    ),
    "listing_table": TemplateSpec(
        "listing_table",
        "Sortable table for larger listing enumerations.",
        "search",
        min_items=1,
    ),
    "scatter_2d": TemplateSpec(
        "scatter_2d",
        "Trade-off between two metrics across barrios.",
        "profile_scatter",
        min_items=3,
    ),
    "none": TemplateSpec(
        "none",
        "No visual; prose stands alone.",
        None,
    ),
}


TEMPLATE_NAMES: list[str] = list(TEMPLATE_REGISTRY)


def template_menu_for_prompt() -> str:
    return "\n".join(
        f"- {spec.name}: {spec.purpose}" for spec in TEMPLATE_REGISTRY.values()
    )
