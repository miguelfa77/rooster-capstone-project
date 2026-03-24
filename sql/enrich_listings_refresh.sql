-- Re-apply derived fields after load_listings (amenities, floor_int, minutes_to_center).
-- Requires columns from sql/migrate_listing_enrichment.sql (or fresh sql/core_tables.sql).

UPDATE core.listings SET
    has_parking = (
        COALESCE(description, '') ~* '(garaje|parking|plaza\s+de\s+parking|garage)'
    ),
    has_terrace = (
        COALESCE(description, '') ~* '(terraza|terrace|balc[oó]n|balcony)'
    ),
    has_elevator = (
        COALESCE(description, '') ~* '(ascensor|elevator|\blift\b)'
    ),
    is_exterior = (
        COALESCE(description, '') ~* 'exterior'
    ),
    is_renovated = (
        COALESCE(description, '') ~* '(reformad|renovad|nuevo\b)'
    ),
    has_ac = (
        COALESCE(description, '') ~* '(aire\s+acondicionado|\ba/c\b|climatizad)'
    ),
    has_storage = (
        COALESCE(description, '') ~* '(trastero|\bstorage\b)'
    );

UPDATE core.listings SET floor_int = CASE
    WHEN floor IS NULL OR btrim(floor) = '' THEN NULL
    WHEN lower(floor) ~ '(^|[^a-z])bajo([^a-z]|$)|\bground\b|planta\s*baja' THEN 0
    WHEN lower(floor) ~ 'semi[\s-]*[aá]tico|entreplanta' THEN 1
    WHEN lower(floor) ~ '[aá]tico|atico|\battic\b' THEN 99
    WHEN floor ~ '[0-9]+' THEN (
        (regexp_match(floor, '([0-9]+)'))[1]
    )::integer
    ELSE NULL
END;

UPDATE core.listings SET minutes_to_center = (
    (regexp_match(COALESCE(time_to_center, ''), '([0-9]+)'))[1]
)::integer
WHERE time_to_center IS NOT NULL AND time_to_center ~ '[0-9]+';

