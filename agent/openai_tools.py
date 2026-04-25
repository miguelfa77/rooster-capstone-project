"""
OpenAI tool schemas for Rooster (Chat + Responses API).
Tools fetch validated data; display composition is handled by RenderPlan.
"""

from __future__ import annotations

from typing import Any


def get_rooster_openai_tools() -> list[dict[str, Any]]:
    """Return the ``tools`` argument for OpenAI tool use (Chat or Responses)."""
    return [
        {
            "type": "function",
            "function": {
                "name": "query_listings",
                "description": (
                    "Query individual property listings in Valencia. Use when the user wants "
                    "specific properties, apartments to buy or rent, underpriced deals, listings "
                    "with amenities, or what is available in an area. "
                    "Do NOT use for barrio-only rankings (use query_neighborhood_profile). "
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
                "name": "query_neighborhood_profile",
                "description": (
                    "Neighborhood rankings and profiles: yield, investment score, price/m², "
                    "comparisons, market overview. Use for barrio comparisons and investment KPIs. "
                    "Set chart_style when user asks for charts. "
                    "Do NOT use for individual listings (use query_listings)."
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
                                "For charts: bar = horizontal bar ranking; "
                                "scatter = yield vs investment score; auto = pick from row count."
                            ),
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
                "name": "query_price_trends",
                "description": (
                    "Listings with price drops or price changes (analytics.price_changes). "
                    "For trend / bajada / subida de precio questions."
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
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "query_chart_data",
                "description": (
                    "Prepare chart data from listings: scatter price vs area, amenity bars, "
                    "€/m² by floor. NOT for VUT density by barrio (use query_neighborhood_density_chart)."
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
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "query_parcel_metrics",
                "description": (
                    "Catastro parcel aggregates by barrio: counts, area distribution (analytics.parcel_metrics). "
                    "Use for build year / parcel stock questions when the user wants structural stock stats."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "neighborhoods": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Filter to these barrio names; empty = all with parcels.",
                        },
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "compare_neighborhoods",
                "description": (
                    "One call returning aligned yield, transport, tourism, listing volume, price "
                    "for several barrios. Use instead of many separate query_neighborhood_profile calls "
                    "when the user compares 2+ barrios on multiple dimensions."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "neighborhoods": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 1,
                            "maxItems": 12,
                            "description": "Barrio names to compare (exact from LIVE list).",
                        },
                        "dimensions": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": [
                                    "yield",
                                    "investment_score",
                                    "tourism",
                                    "transit",
                                    "prices",
                                    "volume",
                                ],
                            },
                            "description": "Optional subset; default = all key dimensions.",
                        },
                    },
                    "required": ["neighborhoods"],
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
        {
            "type": "function",
            "function": {
                "name": "query_neighborhood_density_chart",
                "description": (
                    "VUT / tourist density percentage by barrio for charts (from neighborhood_profile / tourism). "
                    "Use when the user wants density / pressure visual, not a map of VUT points."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "neighborhoods": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Filter; empty = top barrios by tourist_density_pct.",
                        },
                        "metric": {
                            "type": "string",
                            "enum": ["tourist_density_pct", "tourist_apt_count"],
                            "description": "Density % vs raw VUT count.",
                        },
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "query_neighborhood_context",
                "description": (
                    "Phase 4 / optional: qualitative context snippets about a barrio. "
                    "Returns empty when knowledge base is disabled. Use for 'vivir en…' / reputation questions."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "neighborhood": {
                            "type": "string",
                            "description": "Barrio name.",
                        },
                        "topic": {
                            "type": "string",
                            "description": "Optional: schools, night life, family, etc.",
                        },
                    },
                    "required": ["neighborhood"],
                },
            },
        },
    ]
