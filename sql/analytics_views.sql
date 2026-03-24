-- Analytics views for Rooster
-- Run after core_tables.sql, spatial neighborhood_id on listings, migrate_listing_upsert_history if needed.
-- neighborhood_metrics aggregates listings via l.neighborhood_id = n.id (spatial FK), not scraped name text.
-- DROP first: CREATE OR REPLACE VIEW cannot rename/reorder columns vs an existing view (different schema generations).
DROP VIEW IF EXISTS analytics.neighborhood_metrics CASCADE;

CREATE VIEW analytics.neighborhood_metrics AS
SELECT
    n.id AS neighborhood_id,
    n.name AS neighborhood_name,
    n.geom,

    COUNT(l.url) FILTER (WHERE l.operation = 'alquiler')::INTEGER AS alquiler_count,
    ROUND(AVG(l.price_int) FILTER (WHERE l.operation = 'alquiler')::NUMERIC, 0) AS avg_alquiler_price,
    ROUND(MIN(l.price_int) FILTER (WHERE l.operation = 'alquiler')::NUMERIC, 0) AS min_alquiler_price,
    ROUND(MAX(l.price_int) FILTER (WHERE l.operation = 'alquiler')::NUMERIC, 0) AS max_alquiler_price,
    ROUND(
        (PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY l.price_int)
            FILTER (WHERE l.operation = 'alquiler'))::NUMERIC,
        0
    ) AS median_alquiler_price,
    ROUND(AVG(l.area_sqm) FILTER (WHERE l.operation = 'alquiler')::NUMERIC, 1) AS avg_alquiler_area_sqm,
    ROUND(AVG(l.rooms_int) FILTER (WHERE l.operation = 'alquiler')::NUMERIC, 1) AS avg_alquiler_rooms,
    ROUND(
        (PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY (l.price_int::NUMERIC / NULLIF(l.area_sqm, 0)))
            FILTER (WHERE l.operation = 'alquiler' AND l.price_int > 0 AND l.area_sqm > 0))::NUMERIC,
        2
    ) AS median_alquiler_eur_per_sqm,

    COUNT(l.url) FILTER (WHERE l.operation = 'venta')::INTEGER AS venta_count,
    ROUND(AVG(l.price_int) FILTER (WHERE l.operation = 'venta')::NUMERIC, 0) AS avg_venta_price,
    ROUND(MIN(l.price_int) FILTER (WHERE l.operation = 'venta')::NUMERIC, 0) AS min_venta_price,
    ROUND(MAX(l.price_int) FILTER (WHERE l.operation = 'venta')::NUMERIC, 0) AS max_venta_price,
    ROUND(
        (PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY l.price_int)
            FILTER (WHERE l.operation = 'venta'))::NUMERIC,
        0
    ) AS median_venta_price,
    ROUND(AVG(l.area_sqm) FILTER (WHERE l.operation = 'venta')::NUMERIC, 1) AS avg_venta_area_sqm,
    ROUND(AVG(l.rooms_int) FILTER (WHERE l.operation = 'venta')::NUMERIC, 1) AS avg_venta_rooms,
    ROUND(
        (PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY (l.price_int::NUMERIC / NULLIF(l.area_sqm, 0)))
            FILTER (WHERE l.operation = 'venta' AND l.price_int > 0 AND l.area_sqm > 0))::NUMERIC,
        2
    ) AS median_venta_eur_per_sqm,

    COUNT(l.url)::INTEGER AS total_count,

    ROUND(
        (
            12.0
            * (PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY l.price_int)
                FILTER (WHERE l.operation = 'alquiler'))
            / NULLIF(
                (PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY l.price_int)
                    FILTER (WHERE l.operation = 'venta')),
                0
            )
        )::NUMERIC * 100,
        2
    ) AS gross_rental_yield_pct,

    ROUND(
        AVG(
            (l.last_seen_at::date - l.first_seen_at::date)::double precision
        ) FILTER (
            WHERE l.first_seen_at IS NOT NULL AND l.last_seen_at IS NOT NULL
        )::NUMERIC,
        1
    ) AS avg_days_on_market

FROM core.neighborhoods n
LEFT JOIN core.listings l
    ON l.neighborhood_id = n.id
    AND l.price_int > 0
    AND l.area_sqm > 0
GROUP BY n.id, n.name, n.geom;

CREATE OR REPLACE VIEW analytics.listing_summary AS
SELECT
    operation,
    COUNT(*) AS total,
    COUNT(price_int) FILTER (WHERE price_int > 0) AS with_price,
    ROUND(AVG(price_int) FILTER (WHERE price_int > 0)::NUMERIC, 0) AS avg_price,
    ROUND(MIN(price_int) FILTER (WHERE price_int > 0)::NUMERIC, 0) AS min_price,
    ROUND(MAX(price_int) FILTER (WHERE price_int > 0)::NUMERIC, 0) AS max_price,
    ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY price_int) FILTER (WHERE price_int > 0)::NUMERIC, 0) AS median_price,
    ROUND(AVG(area_sqm) FILTER (WHERE area_sqm > 0)::NUMERIC, 1) AS avg_area_sqm,
    ROUND(AVG(rooms_int) FILTER (WHERE rooms_int > 0)::NUMERIC, 1) AS avg_rooms
FROM core.listings
GROUP BY operation;

CREATE OR REPLACE VIEW analytics.price_changes AS
SELECT
    url,
    neighborhood_raw,
    price_int,
    price_int_previous,
    price_int_previous - price_int AS price_drop_eur,
    ROUND(
        ((price_int_previous - price_int)::NUMERIC / NULLIF(price_int_previous, 0) * 100),
        1
    ) AS price_drop_pct,
    last_seen_at
FROM core.listings
WHERE price_int_previous IS NOT NULL
  AND price_int IS NOT NULL
  AND price_int < price_int_previous;
