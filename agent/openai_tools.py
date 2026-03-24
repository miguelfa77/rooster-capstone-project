"""
OpenAI Chat Completions tool schemas for Rooster.
Used with the UI-selected model (no hardcoded model id in call sites).
"""

from __future__ import annotations

from typing import Any

# Meta tool: not executed against the DB; only sets display intent for decide_renderer.
OUTPUT_INTENT_TOOL_NAME = "finalize_output_intent"

# Default renderer before decide_renderer() overrides (matches validate_plan defaults).
DEFAULT_RENDERER_FOR_TOOL: dict[str, str] = {
    "query_listings": "table",
    "query_neighborhood_profile": "bar_chart",
    "query_transit_stops": "transit_map",
    "query_tourist_apartments": "tourism_map",
    "query_price_trends": "table",
    "query_chart_data": "chart",
}


def get_rooster_openai_tools() -> list[dict[str, Any]]:
    """Return the `tools` argument for `client.chat.completions.create`."""
    return [
        {
            "type": "function",
            "function": {
                "name": OUTPUT_INTENT_TOOL_NAME,
                "description": (
                    "REQUIRED whenever you call any data tool in the same turn. Declares how "
                    "the UI should present results (table vs map vs chart, etc.). Call once "
                    "with your batch of data tools — not a database query."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "output_intent": {
                            "type": "string",
                            "enum": [
                                "auto",
                                "table",
                                "map",
                                "map_listings",
                                "map_neighborhoods",
                                "chart",
                                "metrics",
                                "ranking",
                                "combined_map",
                                "cards",
                                "text",
                            ],
                            "description": (
                                "auto: infer from row shapes; table: browsable list; "
                                "map_listings: property dots (lat/lng); map_neighborhoods: barrio polygons; "
                                "map: legacy geographic; chart: Plotly; metrics/cards: KPI cards; "
                                "ranking: bar comparisons; text: prose-only; "
                                "combined_map: listings+transit+tourism overlay."
                            ),
                        }
                    },
                    "required": ["output_intent"],
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
                    "with amenities, or what is available in an area."
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
                        "min_rooms": {"type": "integer"},
                        "max_rooms": {"type": "integer"},
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
                    "required": [],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "query_neighborhood_profile",
                "description": (
                    "Neighborhood rankings and profiles: yield, investment score, price/m², "
                    "comparisons, market overview."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "neighborhoods": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Filter to these names; empty = all neighborhoods.",
                        },
                        "order_by": {
                            "type": "string",
                            "enum": ["investment_score", "yield", "price", "listings"],
                            "description": "Sort order (default investment_score).",
                        },
                        "min_listings": {
                            "type": "integer",
                            "description": "Minimum listings per neighborhood (default 3).",
                        },
                    },
                    "required": [],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "query_transit_stops",
                "description": (
                    "Transit stops (metro, bus) for transport, walkability, connectivity."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "neighborhood": {
                            "type": "string",
                            "description": "Optional barrio filter; omit for city-wide sample.",
                        },
                    },
                    "required": [],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "query_tourist_apartments",
                "description": (
                    "Tourist / VUT / short-term rental pressure (viviendas turísticas)."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "neighborhood": {
                            "type": "string",
                            "description": "Optional barrio filter; omit for city-wide sample.",
                        },
                    },
                    "required": [],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "query_price_trends",
                "description": (
                    "Listings with price drops or price changes (analytics.price_changes)."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "neighborhood": {
                            "type": "string",
                            "description": "Optional barrio filter; omit for city-wide.",
                        },
                        "direction": {
                            "type": "string",
                            "enum": ["down", "both"],
                            "description": "Default both (drops emphasized in DB view).",
                        },
                    },
                    "required": [],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "query_chart_data",
                "description": (
                    "Prepare chart data from listings: scatter price vs area, amenity bars, "
                    "€/m² by floor. UI renders Plotly chart."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "chart_type": {
                            "type": "string",
                            "enum": ["scatter", "amenity", "floor"],
                            "description": "Chart type (default scatter).",
                        },
                    },
                    "required": [],
                },
            },
        },
    ]
