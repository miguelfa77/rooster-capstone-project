"""OpenAI tool schemas for Rooster."""

from __future__ import annotations

from typing import Any

from agent.semantic_layer.sql_builder import AGGREGATIONS, TIME_GRANULARITIES, metric_keys


def get_rooster_openai_tools() -> list[dict[str, Any]]:
    """Return the ``tools`` argument for OpenAI tool use (Chat or Responses)."""
    metric_enum = metric_keys()
    select_metrics_enum = [
        "gross_rental_yield_pct",
        "median_sale",
        "median_alquiler",
        "venta_count",
        "alquiler_count",
        "investment_score",
        "transit_stop_count",
        "tourism_pressure",
        "eur_per_sqm",
        "data_confidence",
    ]
    return [
        {
            "type": "function",
            "function": {
                "name": "select_metrics",
                "description": (
                    "Aggregated neighborhood analytics. Use for rankings, comparisons, market analysis. "
                    "Not for individual listings (use query_listings). metrics: gross_rental_yield_pct, "
                    "median_sale, median_alquiler, venta_count, alquiler_count, investment_score, "
                    "transit_stop_count, tourism_pressure, eur_per_sqm, data_confidence. "
                    "No room/bedroom metrics at neighborhood level. filters use "
                    "[{field, op, value}] with ops: gt, gte, lt, lte, eq, neq, not_null, is_null, in, not_in. "
                    "min_venta_count and min_alquiler_count are top-level integers, not filters. "
                    "include_geometry=true for map output."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "metrics": {
                            "type": "array",
                            "items": {"type": "string", "enum": select_metrics_enum},
                            "minItems": 1,
                            "description": "Neighborhood metrics to return.",
                        },
                        "filters": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "field": {"type": "string"},
                                    "op": {
                                        "type": "string",
                                        "enum": ["gt", "gte", "lt", "lte", "eq", "neq", "not_null", "is_null", "in", "not_in"],
                                    },
                                    "value": {},
                                },
                                "required": ["field", "op"],
                            },
                            "description": "Optional filters as list of field/op/value conditions.",
                        },
                        "order_by": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "field": {"type": "string"},
                                    "direction": {"type": "string", "enum": ["asc", "desc"]},
                                },
                                "required": ["field", "direction"],
                            },
                        },
                        "limit": {"type": "integer", "description": "Max rows, default 20, max 100."},
                        "min_venta_count": {"type": "integer"},
                        "min_alquiler_count": {"type": "integer"},
                        "include_geometry": {"type": "boolean"},
                        "neighborhoods": {"type": "array", "items": {"type": "string"}},
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
