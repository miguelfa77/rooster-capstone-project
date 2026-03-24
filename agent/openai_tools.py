"""
OpenAI Chat Completions tool schemas for Rooster.
Used with the UI-selected model (no hardcoded model id in call sites).

Each data tool requires ``output_intent`` in its parameters so display intent is
declared with the same call as the data request.
"""

from __future__ import annotations

from typing import Any

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
                        "output_intent": {
                            "type": "string",
                            "enum": ["map_listings", "table", "cards", "auto"],
                            "description": (
                                "REQUIRED. How to display results. "
                                "map_listings: user said mapa, map, localización, ubicación, "
                                "enséñame en mapa, show on map, where is. "
                                "table: user wants a list to browse. "
                                "cards: single property or brief summary. "
                                "auto: no clear signal."
                            ),
                        },
                    },
                    "required": ["output_intent"],
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
                        "chart_style": {
                            "type": "string",
                            "enum": ["bar", "scatter", "auto"],
                            "description": (
                                "When showing charts (bar_chart or chart intent): "
                                "bar = horizontal bar ranking (default for many barrios); "
                                "scatter = yield vs investment score (two metrics); "
                                "auto = pick from row count."
                            ),
                        },
                        "output_intent": {
                            "type": "string",
                            "enum": [
                                "map_neighborhoods",
                                "bar_chart",
                                "chart",
                                "table",
                                "cards",
                                "auto",
                            ],
                            "description": (
                                "REQUIRED. How to display results. "
                                "map_neighborhoods: barrio polygons on a map. "
                                "bar_chart or chart: gráfica, chart, graph, visualización, "
                                "rankings, comparisons — use chart_style for bar vs scatter. "
                                "cards: KPI cards. table: detailed list. "
                                "auto: no clear signal."
                            ),
                        },
                    },
                    "required": ["output_intent"],
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
                        "output_intent": {
                            "type": "string",
                            "enum": ["transit_map", "table", "auto"],
                            "description": (
                                "REQUIRED. Almost always transit_map. "
                                "Use table only if the user explicitly asks for a list of stops."
                            ),
                        },
                    },
                    "required": ["output_intent"],
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
                        "output_intent": {
                            "type": "string",
                            "enum": ["tourism_map", "table", "auto"],
                            "description": (
                                "REQUIRED. Almost always tourism_map."
                            ),
                        },
                    },
                    "required": ["output_intent"],
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
                            "enum": ["up", "down", "both"],
                            "description": "Price movement direction (default both).",
                        },
                        "output_intent": {
                            "type": "string",
                            "enum": ["bar_chart", "table", "auto"],
                            "description": "REQUIRED. How to display results.",
                        },
                    },
                    "required": ["output_intent"],
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
                        "output_intent": {
                            "type": "string",
                            "enum": ["chart", "table", "auto"],
                            "description": (
                                "REQUIRED. Usually chart for scatter/amenity/floor visuals."
                            ),
                        },
                    },
                    "required": ["output_intent"],
                },
            },
        },
    ]
