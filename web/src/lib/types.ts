// Contract types: thin aliases over the generated OpenAPI schemas.
import type { components } from "./api-types";

type Schemas = components["schemas"];

export type PrimitivePayload = Schemas["PrimitivePayload"];
/** A block inside a composite (every kind except composite). */
export type InnerPrimitivePayload = Schemas["CompositePayload"]["blocks"][number];
export type PrimitiveKind = PrimitivePayload["kind"];

export type TextPayload = Schemas["TextPayload"];
export type KpiPayload = Schemas["KpiPayload"];
export type TablePayload = Schemas["TablePayload"];
export type TableColumn = Schemas["TableColumn"];
export type CodePayload = Schemas["CodePayload"];
export type ChoroplethPayload = Schemas["ChoroplethPayload"];
export type BarPayload = Schemas["BarPayload"];
export type ScatterPayload = Schemas["ScatterPayload"];
export type LinePayload = Schemas["LinePayload"];
export type PointMapPayload = Schemas["PointMapPayload"];
export type CompositePayload = Schemas["CompositePayload"];
export type TooltipField = Schemas["TooltipField"];

export type ThoughtEvent = Schemas["ThoughtEvent"];
export type TextDeltaEvent = Schemas["TextDeltaEvent"];
export type PrimitiveEvent = Schemas["PrimitiveEvent"];
export type FollowUpsEvent = Schemas["FollowUpsEvent"];
export type ErrorEvent = Schemas["ErrorEvent"];
export type DoneEvent = Schemas["DoneEvent"];
export type TurnEvent = ThoughtEvent | TextDeltaEvent | PrimitiveEvent | FollowUpsEvent | ErrorEvent | DoneEvent;

export type UserMessage = Schemas["UserMessage"];
export type AssistantMessage = Schemas["AssistantMessage"];
export type Message = ConversationOut["messages"][number];
export type ConversationOut = Schemas["ConversationOut"];
export type Fallback = Schemas["Fallback"];

export type MarketFraming = Schemas["MarketFraming"];
export type NeighborhoodsOut = Schemas["NeighborhoodsOut"];
export type RankingRow = Schemas["RankingRow"];
export type YieldLiquidityRow = Schemas["YieldLiquidityRow"];
export type ListingRow = Schemas["ListingRow"];
export type TransitRow = Schemas["TransitRow"];
export type TourismRow = Schemas["TourismRow"];

export type UserOut = Schemas["UserOut"];
/** PATCH /me/preferences body; null or missing keys are left unchanged. */
export type Preferences = Schemas["Preferences"];
export type StoredPreferences = Schemas["StoredPreferences"];
export type ConversationSummary = Schemas["ConversationSummary"];
