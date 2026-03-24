-- Core tables: cleaned, typed data derived from raw
-- Run after bootstrap_rooster.sql

-- 4.1) Core listings (from raw.listings_raw; one row per url — current state)
CREATE TABLE IF NOT EXISTS core.listings (
    url                TEXT PRIMARY KEY,
    operation          TEXT NOT NULL,
    heading            TEXT,
    price              TEXT,
    price_int          INTEGER,
    price_int_previous INTEGER,
    currency           TEXT,
    period             TEXT,
    rooms              TEXT,
    rooms_int          INTEGER,
    area               TEXT,
    area_sqm           DOUBLE PRECISION,
    floor              TEXT,
    time_to_center     TEXT,
    description        TEXT,
    street_name_raw    TEXT,
    neighborhood_raw   TEXT,
    scraped_at         TEXT,
    first_seen_at      TIMESTAMPTZ,
    last_seen_at       TIMESTAMPTZ,
    has_parking        BOOLEAN NOT NULL DEFAULT FALSE,
    has_terrace        BOOLEAN NOT NULL DEFAULT FALSE,
    has_elevator       BOOLEAN NOT NULL DEFAULT FALSE,
    is_exterior        BOOLEAN NOT NULL DEFAULT FALSE,
    is_renovated       BOOLEAN NOT NULL DEFAULT FALSE,
    has_ac             BOOLEAN NOT NULL DEFAULT FALSE,
    has_storage        BOOLEAN NOT NULL DEFAULT FALSE,
    floor_int          INTEGER,
    minutes_to_center  INTEGER,
    lat                DOUBLE PRECISION,
    lng                DOUBLE PRECISION,
    geocode_quality    TEXT,
    neighborhood_id    TEXT,
    days_on_market     INTEGER GENERATED ALWAYS AS (
        CASE
            WHEN first_seen_at IS NOT NULL AND last_seen_at IS NOT NULL
            THEN (last_seen_at::date - first_seen_at::date)
            ELSE NULL
        END
    ) STORED
);

-- 4.1b) Append-only price / presence time series (one row per url per scrape observation)
CREATE TABLE IF NOT EXISTS core.listing_snapshots (
    id         BIGSERIAL PRIMARY KEY,
    url        TEXT NOT NULL,
    price_int  INTEGER,
    scraped_at TIMESTAMPTZ NOT NULL,
    UNIQUE (url, scraped_at)
);
CREATE INDEX IF NOT EXISTS listing_snapshots_url_idx ON core.listing_snapshots (url);

-- 4.2) Core streets (from raw.vias_raw, Valencia city)
CREATE TABLE IF NOT EXISTS core.streets (
    id             SERIAL PRIMARY KEY,
    nombre_municipio TEXT NOT NULL,
    codigo_via     TEXT NOT NULL,
    street_name    TEXT NOT NULL,
    tipo_via       TEXT,
    nombre_via     TEXT,
    UNIQUE (nombre_municipio, codigo_via)
);

-- 4.3) Core parcels (from raw.parcels_raw)
CREATE TABLE IF NOT EXISTS core.parcels (
    refcat    TEXT PRIMARY KEY,
    municipio TEXT,
    via       TEXT,
    numero    TEXT,
    area      DOUBLE PRECISION,
    geom      geometry(POLYGON, 25830)
);

-- 4.4) Core neighborhoods (from raw.neighborhoods_raw)
CREATE TABLE IF NOT EXISTS core.neighborhoods (
    id   TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    geom geometry(MULTIPOLYGON, 4326)
);
