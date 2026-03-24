-- Street + parcel geocoding for raw.tourist_apartments (Valencia city infrastructure).
-- Prerequisites: migrate_listing_enrichment.sql (pg_trgm), core.streets, core.parcels (Valencia city),
--                  core.neighborhoods, raw.tourist_apartments populated by load_tourist_apartments.
-- Coverage is limited to listings that match core.streets / core.parcels (city scope).
-- Run inside a single DB transaction from the Python runner (or wrap in BEGIN/COMMIT in psql).

CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS unaccent;

ALTER TABLE raw.tourist_apartments
    ADD COLUMN IF NOT EXISTS street_parsed TEXT,
    ADD COLUMN IF NOT EXISTS number_parsed TEXT,
    ADD COLUMN IF NOT EXISTS matched_street_code TEXT,
    ADD COLUMN IF NOT EXISTS geocode_quality TEXT,
    ADD COLUMN IF NOT EXISTS neighborhood_id TEXT;

COMMENT ON COLUMN raw.tourist_apartments.geocode_quality IS
    'street = parcel match by via+number; street_centroid = fallback along street; parcel = from ref_catastral loader';

-- Step 1 — Parse address: street = before first comma; number = digits from second segment
UPDATE raw.tourist_apartments
SET
    street_parsed = NULLIF(trim(split_part(coalesce(address, ''), ',', 1)), ''),
    number_parsed = NULLIF(
        trim(regexp_replace(split_part(coalesce(address, ''), ',', 2), '[^0-9]', '', 'g')),
        ''
    )
WHERE address IS NOT NULL;

-- Step 2 — Fuzzy match street to core.streets (Valencia city via table)
UPDATE raw.tourist_apartments ta
SET matched_street_code = sub.codigo_via
FROM (
    SELECT DISTINCT ON (ta.id)
        ta.id,
        s.codigo_via
    FROM raw.tourist_apartments ta
    INNER JOIN core.streets s
        ON ta.street_parsed IS NOT NULL
        AND similarity(
            unaccent(lower(ta.street_parsed)),
            unaccent(lower(s.street_name))
        ) > 0.4
    ORDER BY
        ta.id,
        similarity(
            unaccent(lower(ta.street_parsed)),
            unaccent(lower(s.street_name))
        ) DESC
) AS sub
WHERE ta.id = sub.id;

-- Step 3 — Parcel match by street code + building number (digits only)
WITH parcel_pick AS (
    SELECT DISTINCT ON (ta.id)
        ta.id,
        p.geom
    FROM raw.tourist_apartments ta
    INNER JOIN core.parcels p
        ON p.via = ta.matched_street_code
        AND ta.number_parsed IS NOT NULL
        AND ta.number_parsed <> ''
        AND regexp_replace(trim(coalesce(p.numero, '')), '[^0-9]', '', 'g') = ta.number_parsed
        AND ta.lat IS NULL
    ORDER BY ta.id, p.refcat
)
UPDATE raw.tourist_apartments ta
SET
    lat = ST_Y(ST_Transform(ST_Centroid(ST_MakeValid(pp.geom::geometry)), 4326)),
    lng = ST_X(ST_Transform(ST_Centroid(ST_MakeValid(pp.geom::geometry)), 4326)),
    geocode_quality = 'street'
FROM parcel_pick pp
WHERE ta.id = pp.id;

-- Step 4 — Fallback: centroid of parcels on matched street (no number match).
-- ST_Union can throw GEOS TopologyException on invalid/overlapping parcels; ST_Collect + MakeValid is robust.
WITH street_centroids AS (
    SELECT
        ta.id,
        ST_Y(ST_Transform(ST_Centroid(ST_MakeValid(ST_Collect(ST_MakeValid(p.geom::geometry)))), 4326)) AS lat,
        ST_X(ST_Transform(ST_Centroid(ST_MakeValid(ST_Collect(ST_MakeValid(p.geom::geometry)))), 4326)) AS lng
    FROM raw.tourist_apartments ta
    INNER JOIN core.parcels p ON p.via = ta.matched_street_code
    WHERE ta.lat IS NULL
      AND ta.matched_street_code IS NOT NULL
    GROUP BY ta.id
)
UPDATE raw.tourist_apartments ta
SET
    lat = sc.lat,
    lng = sc.lng,
    geocode_quality = 'street_centroid'
FROM street_centroids sc
WHERE ta.id = sc.id;

-- Step 5 — Neighborhood (same pattern as listings)
UPDATE raw.tourist_apartments ta
SET neighborhood_id = n.id
FROM core.neighborhoods n
WHERE ta.lat IS NOT NULL
  AND ta.lng IS NOT NULL
  AND ta.neighborhood_id IS NULL
  AND ST_Within(
      ST_SetSRID(ST_MakePoint(ta.lng, ta.lat), 4326),
      n.geom
  );

-- Refresh core.tourist_apartments from raw (active + coordinates)
DELETE FROM core.tourist_apartments;

INSERT INTO core.tourist_apartments (
    id, address, license_no, lat, lng, status, geom, neighborhood_id
)
SELECT
    r.id,
    r.address,
    r.license_no,
    r.lat,
    r.lng,
    r.status,
    ST_SetSRID(ST_MakePoint(r.lng, r.lat), 4326),
    r.neighborhood_id
FROM raw.tourist_apartments r
WHERE r.lat IS NOT NULL
  AND r.lng IS NOT NULL
  AND r.status = 'active';

-- Step 6 — Coverage report (run separately if needed)
-- SELECT geocode_quality, COUNT(*) AS n
-- FROM raw.tourist_apartments
-- GROUP BY geocode_quality
-- ORDER BY n DESC;
