-- Street → parcel centroid via core.streets.codigo_via = core.parcels.via; then neighborhood fallback.
-- Run after streets + parcels + listings are loaded. Requires pg_trgm and unaccent.
-- Optional: psql -d rooster -f sql/enrich_geocode.sql (or GEOCODE_LISTINGS=1 python -m pipeline.core.enrich_listings)

CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS unaccent;

ALTER TABLE core.listings ADD COLUMN IF NOT EXISTS geocode_quality TEXT;

-- 1) Drop noise values that are not real street names (saves geocoding work)
UPDATE core.listings
SET street_name_raw = NULL
WHERE street_name_raw ~ '^\d+$'
   OR lower(trim(street_name_raw)) = 's/n'
   OR length(trim(street_name_raw)) < 4;

-- 2) Reset neighborhood-only points so we can assign street-level coords where possible
UPDATE core.listings
SET lat = NULL, lng = NULL, geocode_quality = NULL
WHERE geocode_quality = 'neighborhood';

-- 3) Abbreviate full Spanish/Valencian forms to match core.streets (e.g. AV BLASCO IBANEZ)
UPDATE core.listings
SET street_name_raw = btrim(
    regexp_replace(
        unaccent(
            regexp_replace(
                regexp_replace(
                    regexp_replace(
                        regexp_replace(upper(street_name_raw),
                            'AVENIDA DE (LA |LOS |EL |L''|LES |DE )*', 'AV '),
                        'CALLE DE (LA |LOS |EL |L''|LES |DE )*', 'CL '),
                    'PASEO DE (LA |L'')*', 'PS '),
                'PLAZA DE (LA |LOS |EL |L'')*', 'PZ ')
        ),
        '\s+', ' ', 'g'
    )
)
WHERE street_name_raw IS NOT NULL
  AND lat IS NULL;

-- 4) Street geocode: best street_name by similarity, parcel join p.via = s.codigo_via
WITH cand AS (
    SELECT DISTINCT ON (l.url)
        l.url,
        p.geom
    FROM core.listings l
    INNER JOIN core.streets s
        ON s.street_name = (
            SELECT s2.street_name
            FROM core.streets s2
            WHERE similarity(
                unaccent(lower(btrim(l.street_name_raw))),
                unaccent(lower(s2.street_name))
            ) > 0.3
            ORDER BY similarity(
                unaccent(lower(btrim(l.street_name_raw))),
                unaccent(lower(s2.street_name))
            ) DESC
            LIMIT 1
        )
    INNER JOIN core.parcels p
        ON p.via IS NOT NULL
        AND btrim(p.via) <> ''
        AND p.via = s.codigo_via
    WHERE l.geocode_quality IS DISTINCT FROM 'street'
      AND l.street_name_raw IS NOT NULL
      AND btrim(l.street_name_raw) <> ''
    ORDER BY l.url, p.refcat
)
UPDATE core.listings l
SET
    lat = ST_Y(ST_Transform(ST_Centroid(c.geom), 4326)),
    lng = ST_X(ST_Transform(ST_Centroid(c.geom), 4326)),
    geocode_quality = 'street'
FROM cand c
WHERE l.url = c.url;

-- 5) Neighborhood centroid fallback (best name match per listing)
WITH nh AS (
    SELECT DISTINCT ON (l.url)
        l.url,
        n.geom
    FROM core.listings l
    INNER JOIN core.neighborhoods n
        ON l.lat IS NULL
        AND l.neighborhood_raw IS NOT NULL
        AND btrim(l.neighborhood_raw) <> ''
        AND similarity(
            unaccent(upper(trim(l.neighborhood_raw))),
            unaccent(upper(trim(n.name)))
        ) > 0.4
    ORDER BY
        l.url,
        similarity(
            unaccent(upper(trim(l.neighborhood_raw))),
            unaccent(upper(trim(n.name)))
        ) DESC
)
UPDATE core.listings l
SET
    lat = ST_Y(ST_Transform(ST_Centroid(nh.geom), 4326)),
    lng = ST_X(ST_Transform(ST_Centroid(nh.geom), 4326)),
    geocode_quality = 'neighborhood'
FROM nh
WHERE l.url = nh.url
  AND l.lat IS NULL;

-- 6) Quality mix (run in psql to inspect; harmless if executed from app)
SELECT geocode_quality,
       COUNT(*) AS listings,
       ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 1) AS pct
FROM core.listings
GROUP BY geocode_quality
ORDER BY listings DESC;
