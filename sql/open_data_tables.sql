-- Open data: transit stops + tourist apartments (Valencia).
-- Run after: bootstrap_rooster.sql, core_tables.sql, neighborhoods + listings loaded,
--            match_listings_neighborhood_spatial.sql (recommended).
-- Then run `dbt run` from dbt/ for analytics.* views (see README). Re-run UPDATE block after loaders.

-- ---------------------------------------------------------------------------
-- raw.transit_stops — staging from Overpass / EMT exports
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS raw.transit_stops (
    osm_id    BIGINT PRIMARY KEY,
    name      TEXT,
    stop_type TEXT,
    lat       DOUBLE PRECISION,
    lng       DOUBLE PRECISION,
    source    TEXT DEFAULT 'overpass',
    loaded_at TIMESTAMPTZ DEFAULT now()
);

-- ---------------------------------------------------------------------------
-- raw.tourist_apartments — staging from Valencia open data CSV
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS raw.tourist_apartments (
    id         TEXT PRIMARY KEY,
    address    TEXT,
    license_no TEXT,
    lat        DOUBLE PRECISION,
    lng        DOUBLE PRECISION,
    status     TEXT,
    loaded_at  TIMESTAMPTZ DEFAULT now()
);

ALTER TABLE raw.tourist_apartments
    ADD COLUMN IF NOT EXISTS street_parsed TEXT,
    ADD COLUMN IF NOT EXISTS number_parsed TEXT,
    ADD COLUMN IF NOT EXISTS matched_street_code TEXT,
    ADD COLUMN IF NOT EXISTS geocode_quality TEXT,
    ADD COLUMN IF NOT EXISTS neighborhood_id TEXT;

-- ---------------------------------------------------------------------------
-- core.transit_stops — points + barrio assignment
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS core.transit_stops (
    osm_id            BIGINT PRIMARY KEY,
    name              TEXT,
    stop_type         TEXT,
    lat               DOUBLE PRECISION NOT NULL,
    lng               DOUBLE PRECISION NOT NULL,
    geom              geometry(POINT, 4326) NOT NULL,
    neighborhood_id TEXT
);

CREATE INDEX IF NOT EXISTS idx_core_transit_stops_geom
    ON core.transit_stops USING GIST (geom);
CREATE INDEX IF NOT EXISTS idx_core_transit_stops_neighborhood
    ON core.transit_stops (neighborhood_id);

-- ---------------------------------------------------------------------------
-- core.tourist_apartments — active licenses with geometry
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS core.tourist_apartments (
    id                TEXT PRIMARY KEY,
    address           TEXT,
    license_no        TEXT,
    lat               DOUBLE PRECISION NOT NULL,
    lng               DOUBLE PRECISION NOT NULL,
    geom              geometry(POINT, 4326) NOT NULL,
    status            TEXT,
    neighborhood_id   TEXT
);

CREATE INDEX IF NOT EXISTS idx_core_tourist_apartments_geom
    ON core.tourist_apartments USING GIST (geom);
CREATE INDEX IF NOT EXISTS idx_core_tourist_apartments_neighborhood
    ON core.tourist_apartments (neighborhood_id);

-- ---------------------------------------------------------------------------
-- core.listings — distance to nearest transit stop (metres)
-- ---------------------------------------------------------------------------
ALTER TABLE core.listings
    ADD COLUMN IF NOT EXISTS nearest_stop_m INTEGER;

COMMENT ON COLUMN core.listings.nearest_stop_m IS
    'Metres to nearest core.transit_stops (geodesic); NULL if no lat/lng or no stops.';

-- Refresh after loading core.transit_stops (re-run this block anytime)
UPDATE core.listings
SET nearest_stop_m = NULL
WHERE NOT EXISTS (SELECT 1 FROM core.transit_stops LIMIT 1);

-- Correlated subquery: UPDATE target cannot be referenced inside LATERAL in FROM.
UPDATE core.listings l
SET nearest_stop_m = (
    SELECT ROUND(
        ST_Distance(
            ST_SetSRID(ST_MakePoint(l.lng, l.lat), 4326)::geography,
            t.geom::geography
        )
    )::integer
    FROM core.transit_stops t
    ORDER BY ST_SetSRID(ST_MakePoint(l.lng, l.lat), 4326) <-> t.geom
    LIMIT 1
)
WHERE l.lat IS NOT NULL
  AND l.lng IS NOT NULL
  AND EXISTS (SELECT 1 FROM core.transit_stops LIMIT 1);

UPDATE core.listings
SET nearest_stop_m = NULL
WHERE lat IS NULL OR lng IS NULL;
