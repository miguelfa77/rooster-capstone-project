"""Response models the API publishes in api/openapi.json for client type generation.

Primitive payloads are the synthesizer primitives as `build_primitive_payload` makes
them self-contained: descriptors carry their `data` rows, `code` carries the sandbox
output. Message and payload models ignore unknown keys, so stored
messages can keep internal fields (execution results, SQL) that the API does not send.
Runtime behaviour does not depend on these models except GET /conversations/{id},
which filters stored messages through ConversationOut, and the user, preference and
rename request bodies, which validate input.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator

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
# Each payload subclasses its synthesizer primitive (one source of field definitions)
# and ignores extra keys, so one odd stored block cannot break GET /conversations/{id}.


class TextPayload(TextPrimitive):
    model_config = _IGNORE_EXTRA


class KpiPayload(KpiPrimitive):
    model_config = _IGNORE_EXTRA


class TablePayload(TablePrimitive):
    model_config = _IGNORE_EXTRA


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
        TextPayload,
        KpiPayload,
        TablePayload,
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
        TextPayload,
        KpiPayload,
        TablePayload,
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
    title: str | None
    messages: list[Message]


class ConversationSummary(BaseModel):
    id: uuid.UUID
    title: str | None
    updated_at: datetime


class ConversationRename(BaseModel):
    title: str

    @field_validator("title")
    @classmethod
    def _trimmed_1_to_120(cls, v: str) -> str:
        v = v.strip()
        if not 1 <= len(v) <= 120:
            raise ValueError("title must be 1-120 characters after trimming")
        return v


# ── Users ─────────────────────────────────────────────────────────────────


class Preferences(BaseModel):
    """PATCH /me/preferences body. Unknown keys are rejected; null or missing keys are left unchanged."""

    model_config = ConfigDict(extra="forbid")
    theme: Literal["light", "dark", "system"] | None = None
    chat_open: bool | None = None
    ticker_visible: bool | None = None


class StoredPreferences(Preferences):
    """Preferences as returned: stray stored keys are dropped instead of failing the response."""

    model_config = ConfigDict(extra="ignore")


class UserIn(BaseModel):
    email: str
    name: str | None = None
    image: str | None = None

    @field_validator("email")
    @classmethod
    def _looks_like_email(cls, v: str) -> str:
        v = v.strip()
        if "@" not in v:
            raise ValueError("email must contain @")
        return v


class UserOut(BaseModel):
    id: uuid.UUID
    email: str
    name: str | None
    image: str | None
    preferences: StoredPreferences


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
