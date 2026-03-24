-- Listing enrichment: amenities, floor/time parsing, days on market, lat/lng, trgm indexes.
-- Run from repo root: psql -d rooster -f sql/migrate_listing_enrichment.sql
-- Then refresh analytics (avg_days_on_market uses date diff on listings): psql -d rooster -f sql/analytics_views.sql
--
-- Requires PostGIS. Enables pg_trgm for fuzzy street → parcel geocoding.

CREATE EXTENSION IF NOT EXISTS pg_trgm;

ALTER TABLE core.listings ADD COLUMN IF NOT EXISTS has_parking BOOLEAN NOT NULL DEFAULT FALSE;
ALTER TABLE core.listings ADD COLUMN IF NOT EXISTS has_terrace BOOLEAN NOT NULL DEFAULT FALSE;
ALTER TABLE core.listings ADD COLUMN IF NOT EXISTS has_elevator BOOLEAN NOT NULL DEFAULT FALSE;
ALTER TABLE core.listings ADD COLUMN IF NOT EXISTS is_exterior BOOLEAN NOT NULL DEFAULT FALSE;
ALTER TABLE core.listings ADD COLUMN IF NOT EXISTS is_renovated BOOLEAN NOT NULL DEFAULT FALSE;
ALTER TABLE core.listings ADD COLUMN IF NOT EXISTS has_ac BOOLEAN NOT NULL DEFAULT FALSE;
ALTER TABLE core.listings ADD COLUMN IF NOT EXISTS has_storage BOOLEAN NOT NULL DEFAULT FALSE;

ALTER TABLE core.listings ADD COLUMN IF NOT EXISTS floor_int INTEGER;
ALTER TABLE core.listings ADD COLUMN IF NOT EXISTS minutes_to_center INTEGER;
ALTER TABLE core.listings ADD COLUMN IF NOT EXISTS lat DOUBLE PRECISION;
ALTER TABLE core.listings ADD COLUMN IF NOT EXISTS lng DOUBLE PRECISION;
ALTER TABLE core.listings ADD COLUMN IF NOT EXISTS geocode_quality TEXT;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM information_schema.columns
        WHERE table_schema = 'core'
          AND table_name = 'listings'
          AND column_name = 'days_on_market'
    ) THEN
        ALTER TABLE core.listings ADD COLUMN days_on_market INTEGER GENERATED ALWAYS AS (
            CASE
                WHEN first_seen_at IS NOT NULL AND last_seen_at IS NOT NULL
                THEN (last_seen_at::date - first_seen_at::date)
                ELSE NULL
            END
        ) STORED;
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS streets_street_name_trgm ON core.streets USING gin (street_name gin_trgm_ops);
CREATE INDEX IF NOT EXISTS streets_nombre_via_trgm ON core.streets USING gin (nombre_via gin_trgm_ops)
    WHERE nombre_via IS NOT NULL AND btrim(nombre_via) <> '';
CREATE INDEX IF NOT EXISTS parcels_via_trgm ON core.parcels USING gin (via gin_trgm_ops)
    WHERE via IS NOT NULL AND btrim(via) <> '';
CREATE INDEX IF NOT EXISTS listings_street_raw_trgm ON core.listings USING gin (street_name_raw gin_trgm_ops)
    WHERE street_name_raw IS NOT NULL AND btrim(street_name_raw) <> '';

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

-- Geocoding (optional, slower): psql -d rooster -f sql/enrich_geocode.sql
-- Or: GEOCODE_LISTINGS=1 python -m pipeline.core.enrich_listings
