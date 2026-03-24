-- Assign core.listings.neighborhood_id from lat/lng via ST_Within(point, barrio polygon).
-- Requires PostGIS, core.neighborhoods.geom (MULTIPOLYGON, 4326), listings with coordinates.
-- Run: psql -d rooster -f sql/match_listings_neighborhood_spatial.sql

ALTER TABLE core.listings ADD COLUMN IF NOT EXISTS neighborhood_id TEXT;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'listings_neighborhood_id_fkey'
    ) THEN
        ALTER TABLE core.listings
            ADD CONSTRAINT listings_neighborhood_id_fkey
            FOREIGN KEY (neighborhood_id) REFERENCES core.neighborhoods(id);
    END IF;
EXCEPTION
    WHEN duplicate_object THEN NULL;
END $$;

-- Spatial join: point must lie inside neighborhood polygon (geometrically exact in CRS).
UPDATE core.listings l
SET neighborhood_id = (
    SELECT n.id
    FROM core.neighborhoods n
    WHERE n.geom IS NOT NULL
      AND ST_Within(
          ST_SetSRID(ST_MakePoint(l.lng, l.lat), 4326)::geometry,
          n.geom
      )
    LIMIT 1
)
WHERE l.lat IS NOT NULL
  AND l.lng IS NOT NULL
  AND l.neighborhood_id IS NULL;

-- Coverage snapshot
SELECT
    COUNT(*) FILTER (WHERE neighborhood_id IS NOT NULL) AS spatially_matched,
    COUNT(*) FILTER (WHERE neighborhood_id IS NULL AND lat IS NOT NULL) AS geocoded_unmatched,
    COUNT(*) FILTER (WHERE lat IS NULL) AS no_coords,
    COUNT(*) AS total
FROM core.listings;
