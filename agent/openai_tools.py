"""OpenAI tool schemas for Rooster."""

from __future__ import annotations

from typing import Any

from agent.semantic_layer.sql_builder import AGGREGATIONS, TIME_GRANULARITIES, metric_keys


def get_rooster_openai_tools() -> list[dict[str, Any]]:
    """Return the ``tools`` argument for OpenAI tool use (Chat or Responses)."""
    metric_enum = metric_keys()
    return [
        {
            "type": "function",
            "function": {
                "name": "select_metrics",
                "description": (
                    "Compositional analytical workhorse. Use for neighborhood rankings, comparisons, "
                    "filters, and ordered metric tables. NOT for individual property listings. "
                    "Use it for metric questions. Metrics must come from the semantic registry. "
                    "Default sample filter is venta>=3 and alquiler>=3; "
                    "set min_listings {venta:0, alquiler:0} only when the user explicitly asks to include thin samples."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "metrics": {
                            "type": "array",
                            "items": {"type": "string", "enum": metric_enum},
                            "minItems": 1,
                            "description": "Canonical metrics to return.",
                        },
                        "filters": {
                            "type": "object",
                            "description": (
                                "Optional filters keyed by canonical metric. Each value: "
                                "{operator: >=|<=|>|<|=|!=|in|not_in, value: number|string|array}."
                            ),
                        },
                        "group_by": {
                            "type": "array",
                            "items": {"type": "string", "enum": ["neighborhood"]},
                            "description": "Currently only neighborhood is supported.",
                        },
                        "order_by": {
                            "type": "object",
                            "properties": {
                                "metric": {"type": "string", "enum": metric_enum},
                                "direction": {"type": "string", "enum": ["asc", "desc"]},
                            },
                        },
                        "limit": {"type": "integer", "description": "Max rows, default 20."},
                        "min_listings": {
                            "type": "object",
                            "properties": {
                                "venta": {"type": "integer"},
                                "alquiler": {"type": "integer"},
                            },
                            "description": "Sample-size minima, default {venta:3, alquiler:3}.",
                        },
                        "neighborhoods": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional exact barrio names to restrict the result.",
                        },
                    },
                    "required": ["metrics"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "compute_aggregate",
                "description": (
                    "Compute one summary statistic for a canonical metric after optional filters. "
                    "Use for questions like city median, average, p25/p75, min/max, count, or spread."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "metric": {"type": "string", "enum": metric_enum},
                        "filters": {
                            "type": "object",
                            "description": "Optional metric filters using the same shape as select_metrics.",
                        },
                        "aggregation": {
                            "type": "string",
                            "enum": sorted(AGGREGATIONS),
                        },
                        "min_listings": {
                            "type": "object",
                            "properties": {
                                "venta": {"type": "integer"},
                                "alquiler": {"type": "integer"},
                            },
                        },
                    },
                    "required": ["metric", "aggregation"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "temporal_series",
                "description": (
                    "Return time series over listing snapshots for trends. Use for price/rent/count "
                    "over time. Not for static rankings; use select_metrics for those."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "metric": {
                            "type": "string",
                            "enum": [
                                "median_venta_price",
                                "median_alquiler_price",
                                "venta_count",
                                "alquiler_count",
                            ],
                        },
                        "filters": {
                            "type": "object",
                            "description": (
                                "Optional filters. For neighborhoods use "
                                "{neighborhood_name: {operator: in, value: [...]}}."
                            ),
                        },
                        "time_granularity": {
                            "type": "string",
                            "enum": sorted(TIME_GRANULARITIES),
                        },
                    },
                    "required": ["metric", "time_granularity"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "query_listings",
                "description": (
                    "Query individual property listings in Valencia. Use when the user wants "
                    "specific properties, apartments to buy or rent, underpriced deals, listings "
                    "with amenities, or what is available in an area. "
                    "Do NOT use for barrio-only rankings or comparisons (use select_metrics). "
                    "Common errors: wrong operation (venta vs alquiler) — follow RESOLVED INTENT."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "neighborhood": {
                            "type": "string",
                            "description": (
                                "Neighborhood name; match the LIVE DATABASE STATE list when possible."
                            ),
                        },
                        "operation": {
                            "type": "string",
                            "enum": ["venta", "alquiler", "both"],
                            "description": "Sale, rental, or both (default venta if omitted).",
                        },
                        "max_price": {
                            "type": "integer",
                            "description": "Maximum price in euros.",
                        },
                        "min_price": {
                            "type": "integer",
                            "description": "Minimum price in euros.",
                        },
                        "min_rooms": {
                            "type": "integer",
                            "description": (
                                "Minimum bedrooms. If the user asks for an **exact** number "
                                "(e.g. '2 habitaciones', 'piso de 3 dormitorios', 'con 2 hab.'), "
                                "set **both** min_rooms **and** max_rooms to that same number. "
                                "Use only min_rooms (leave max_rooms unset) when they say "
                                "'al menos', 'mínimo', 'at least', or 'más de N habitaciones'."
                            ),
                        },
                        "max_rooms": {
                            "type": "integer",
                            "description": (
                                "Maximum bedrooms. For an **exact** room count, set max_rooms "
                                "equal to min_rooms (same integer)."
                            ),
                        },
                        "only_below_median": {
                            "type": "boolean",
                            "description": "Only listings priced below neighborhood median.",
                        },
                        "amenities": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": [
                                    "parking",
                                    "terrace",
                                    "elevator",
                                    "ac",
                                    "renovated",
                                ],
                            },
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Max rows (1–100, default 25).",
                        },
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "query_transit_stops",
                "description": (
                    "Transit stops (metro, bus) for transport, walkability, connectivity. "
                    "Prefer this over listings when the user only asks for transport / metro / bus map."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "neighborhood": {
                            "type": "string",
                            "description": "Optional barrio filter; omit for city-wide sample.",
                        },
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "query_tourist_apartments",
                "description": (
                    "Tourist / VUT / short-term rental pressure (viviendas turísticas). "
                    "Not for long-term rental listings — those are in query_listings (alquiler)."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "neighborhood": {
                            "type": "string",
                            "description": "Optional barrio filter; omit for city-wide sample.",
                        },
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "resolve_spatial_reference",
                "description": (
                    "Map qualitative place phrases (e.g. centro, cerca de la playa) to barrio name lists. "
                    "Call before filtering listings when the user did not name specific barrios."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reference": {
                            "type": "string",
                            "description": "Short phrase, Spanish (e.g. cerca de la playa, zona universitaria).",
                        },
                    },
                    "required": ["reference"],
                },
            },
        },
    ]
