"""Typed models for the deterministic semantic resolver."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

Criticality = Literal["essential", "flavour"]


class MetricEntry(BaseModel):
    key: str
    column: str
    spanish: list[str] = Field(default_factory=list)
    english: list[str] = Field(default_factory=list)
    display: str
    formula: str | None = None
    components: list[str] = Field(default_factory=list)
    unit: str | None = None
    caveat: str | None = None
    criticality: Criticality = "essential"
    value_range: dict[str, float | int | None] = Field(default_factory=dict)


class ConceptEntry(BaseModel):
    key: str
    terms: list[str] = Field(default_factory=list)
    expression: dict[str, Any] = Field(default_factory=dict)
    display: str
    criticality: Criticality = "essential"
    variants: list[str] = Field(default_factory=list)


class HeuristicEntry(BaseModel):
    key: str
    terms: list[str] = Field(default_factory=list)
    barrio_ids: list[str] = Field(default_factory=list)
    barrio_names: list[str] = Field(default_factory=list)
    geometry: str | None = None
    display: str
    criticality: Criticality = "flavour"


class EssentialTermEntry(BaseModel):
    key: str
    terms: list[str] = Field(default_factory=list)
    criticality: Criticality = "essential"


class SemanticRegistry(BaseModel):
    metrics: list[MetricEntry] = Field(default_factory=list)
    concepts: list[ConceptEntry] = Field(default_factory=list)
    heuristics: list[HeuristicEntry] = Field(default_factory=list)
    essential_terms: list[EssentialTermEntry] = Field(default_factory=list)


class MatchSpan(BaseModel):
    start: int
    end: int
    text: str
    registry: Literal["metrics", "concepts", "heuristics"]
    key: str
    criticality: Criticality


class ResolvedMetric(BaseModel):
    key: str
    column: str
    display: str
    unit: str | None = None
    formula: str | None = None
    components: list[str] = Field(default_factory=list)
    caveat: str | None = None
    matched_text: str


class ResolvedConcept(BaseModel):
    key: str
    display: str
    expression: dict[str, Any] = Field(default_factory=dict)
    variants: list[str] = Field(default_factory=list)
    matched_text: str


class ResolvedHeuristic(BaseModel):
    key: str
    display: str
    barrio_ids: list[str] = Field(default_factory=list)
    barrio_names: list[str] = Field(default_factory=list)
    geometry: str | None = None
    matched_text: str


class LiteralValue(BaseModel):
    kind: Literal["currency", "rooms", "number", "neighborhood"]
    value: Any
    raw: str
    unit: str | None = None


class ResolvedQuery(BaseModel):
    original_text: str
    resolved_metrics: list[ResolvedMetric] = Field(default_factory=list)
    resolved_concepts: list[ResolvedConcept] = Field(default_factory=list)
    resolved_heuristics: list[ResolvedHeuristic] = Field(default_factory=list)
    literals: list[LiteralValue] = Field(default_factory=list)
    unresolved_essential_terms: list[str] = Field(default_factory=list)
    unresolved_flavour_terms: list[str] = Field(default_factory=list)
    match_spans: list[MatchSpan] = Field(default_factory=list)

    @property
    def needs_clarification(self) -> bool:
        return bool(self.unresolved_essential_terms)
