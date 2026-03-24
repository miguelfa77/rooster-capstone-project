-- Bootstrap script for Rooster analytical database
-- Assumes you are already connected to the `rooster` database in psql:
--   createdb rooster
--   psql -d rooster -f sql/bootstrap_rooster.sql

-- 1) Enable PostGIS (geospatial support)
CREATE EXTENSION IF NOT EXISTS postgis;

-- 2) Create logical schemas
CREATE SCHEMA IF NOT EXISTS raw;
CREATE SCHEMA IF NOT EXISTS core;
CREATE SCHEMA IF NOT EXISTS analytics;

-- 3) Raw tables mirror the source data as closely as possible

-- 3.1) Idealista raw listings (from idealista_alquiler/venta.csv)
-- One row per (url, scraped_at) so each scrape run can append history for time series.
CREATE TABLE IF NOT EXISTS raw.listings_raw (
    operation      TEXT,
    heading        TEXT,
    price          TEXT,
    currency       TEXT,
    period         TEXT,
    rooms          TEXT,
    area           TEXT,
    floor          TEXT,
    time_to_center TEXT,
    description    TEXT,
    url            TEXT NOT NULL,
    page           TEXT,
    scraped_at     TEXT NOT NULL,
    PRIMARY KEY (url, scraped_at)
);

-- 3.2) Catastro parcels raw (from PARCELA.shp DBF + geometry)
-- Geometry SRID: use 25830 (ETRS89 / UTM zone 30N) to match Spanish cadastre convention,
-- or 4326 if you reproject on load. We pick 25830 here.
CREATE TABLE IF NOT EXISTS raw.parcels_raw (
    refcat   TEXT,
    municipio TEXT,
    masa     TEXT,
    parcela  TEXT,
    via      TEXT,
    numero   TEXT,
    area     DOUBLE PRECISION,
    coorx    DOUBLE PRECISION,
    coory    DOUBLE PRECISION,
    geom     geometry(POLYGON, 25830)
);

-- 3.3) Catastro vías raw (from catastro_vias.csv)
CREATE TABLE IF NOT EXISTS raw.vias_raw (
    provincia        TEXT,
    nombre_municipio TEXT,
    codigo_via       TEXT,
    tipo_via         TEXT,
    nombre_via       TEXT,
    via_code         TEXT,
    street_name      TEXT
);

-- 3.4) Neighborhoods raw (from barris-barrios.geojson)
-- We store geometry in SRID 4326 for compatibility with most GIS tools.
CREATE TABLE IF NOT EXISTS raw.neighborhoods_raw (
    id        TEXT,
    name      TEXT,
    geom      geometry(MULTIPOLYGON, 4326)
);

-- Next: run sql/core_tables.sql to create core schema, then pipeline.core.run_all

