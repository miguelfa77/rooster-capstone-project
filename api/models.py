"""Response models the API publishes in api/openapi.json for client type generation.

Primitive payloads are the synthesizer primitives as `build_primitive_payload` makes
them self-contained: descriptors carry their `data` rows, `code` carries the sandbox
output. Message models and the enriched payloads ignore unknown keys, so stored
messages can keep internal fields (execution results, SQL) that the API does not send.
Runtime behaviour does not depend on these models except GET /conversations/{id},
which filters stored messages through ConversationOut.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field

from agent.synthesizer import (
    BarDescriptor,
    ChoroplethDescriptor,
    CodePrimitive,
    CompositePrimitive,
    KpiPrimitive,
    LineDescriptor,
    PointMapDescriptor,
    ScatterDescriptor,
    TablePrimitive,
    TextPrimitive,
)

_IGNORE_EXTRA = ConfigDict(extra="ignore")
Rows = list[dict[str, Any]]


# ── Primitive payloads ────────────────────────────────────────────────────
# Text, kpi and table payloads are the synthesizer primitives unchanged (strict, as
# build_primitive_payload adds nothing to them); the rest add fields and tolerate extras.


class CodePayload(CodePrimitive):
    model_config = _IGNORE_EXTRA
    output_type: Literal["plotly_json", "folium_html"]
    output: str


class ChoroplethPayload(ChoroplethDescriptor):
    model_config = _IGNORE_EXTRA
    data: Rows = []


class BarPayload(BarDescriptor):
    model_config = _IGNORE_EXTRA
    data: Rows = []


class ScatterPayload(ScatterDescriptor):
    model_config = _IGNORE_EXTRA
    data: Rows = []


class LinePayload(LineDescriptor):
    model_config = _IGNORE_EXTRA
    data: Rows = []


class PointMapPayload(PointMapDescriptor):
    model_config = _IGNORE_EXTRA
    data: Rows = []


InnerPrimitivePayload = Annotated[
    Union[
        TextPrimitive,
        KpiPrimitive,
        TablePrimitive,
        CodePayload,
        ChoroplethPayload,
        BarPayload,
        ScatterPayload,
        LinePayload,
        PointMapPayload,
    ],
    Field(discriminator="kind"),
]


class CompositePayload(CompositePrimitive):
    model_config = _IGNORE_EXTRA
    blocks: list[InnerPrimitivePayload] = []


PrimitivePayload = Annotated[
    Union[
        TextPrimitive,
        KpiPrimitive,
        TablePrimitive,
        CodePayload,
        ChoroplethPayload,
        BarPayload,
        ScatterPayload,
        LinePayload,
        PointMapPayload,
        CompositePayload,
    ],
    Field(discriminator="kind"),
]


# ── Conversations ─────────────────────────────────────────────────────────


class Fallback(BaseModel):
    reason: str
    suggestion: str


class UserMessage(BaseModel):
    role: Literal["user"]
    content: str


class AssistantMessage(BaseModel):
    role: Literal["assistant"]
    blocks: list[PrimitivePayload] = []
    summary: str = ""
    primitive_payloads: list[PrimitivePayload] = []
    follow_ups: list[str] = []
    graceful_fallback: Fallback | None = None


Message = Annotated[Union[UserMessage, AssistantMessage], Field(discriminator="role")]


class ConversationOut(BaseModel):
    id: str
    messages: list[Message]


# ── Intel ─────────────────────────────────────────────────────────────────
# Columns as agent/intel_data.py returns them; all nullable, extras ignored.


class MarketFraming(BaseModel):
    city_median_yield: float | None = None
    spread_pp: float | None = None
    n_strong: int | None = None
    n_with_yield: int | None = None


class NeighborhoodsOut(BaseModel):
    geojson: dict[str, Any]
    vmin: float | None
    vmax: float | None


class TransitRow(BaseModel):
    lat: float | None = None
    lng: float | None = None
    name: str | None = None
    stop_type: str | None = None


class TourismRow(BaseModel):
    lat: float | None = None
    lng: float | None = None
    address: str | None = None


class ListingRow(BaseModel):
    url: str | None = None
    operation: str | None = None
    price_int: float | None = None
    area_sqm: float | None = None
    rooms_int: float | None = None
    floor_int: float | None = None
    lat: float | None = None
    lng: float | None = None
    geocode_quality: str | None = None
    neighborhood_id: int | None = None
    neighborhood_name: str | None = None
    has_parking: bool | None = None
    has_terrace: bool | None = None
    has_elevator: bool | None = None
    is_exterior: bool | None = None
    is_renovated: bool | None = None
    has_ac: bool | None = None
    nearest_stop_m: float | None = None
    eur_per_sqm: float | None = None
    neighborhood_median_sqm: float | None = None
    neighborhood_median_sale: float | None = None
    neighborhood_median_rent: float | None = None
    yield_pct: float | None = None
    below_median: bool | None = None


class RankingRow(BaseModel):
    neighborhood_name: str | None = None
    yield_pct: float | None = None
    eur_per_sqm: float | None = None
    transit_stop_count: int | None = None
    transport_rating: str | None = None
    tourist_pct: float | None = None
    tourism_pressure: str | None = None
    score: float | None = None
    venta_count: int | None = None
    alquiler_count: int | None = None


class YieldLiquidityRow(BaseModel):
    neighborhood_name: str | None = None
    yield_pct: float | None = None
    listing_count: int | None = None
    investment_score: float | None = None
    transport_rating: str | None = None
    tourism_pressure: str | None = None
    eur_per_sqm: float | None = None
