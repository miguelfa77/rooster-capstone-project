"""
Keyword families used to derive boolean amenity flags from listing descriptions.

Source of truth for population is ``sql/enrich_listings_refresh.sql`` (regex).
Keep this dict aligned when adding patterns.
"""

AMENITY_PATTERNS: dict[str, list[str]] = {
    "has_parking": ["garaje", "parking", "plaza de parking", "garage"],
    "has_terrace": ["terraza", "terrace", "balcón", "balcon", "balcony"],
    "has_elevator": ["ascensor", "elevator", "lift"],
    "is_exterior": ["exterior", "exteriores"],
    "is_renovated": ["reformado", "renovado", "reformada", "nuevo"],
    "has_ac": ["aire acondicionado", "a/c", "climatizado"],
    "has_storage": ["trastero", "storage"],
}
