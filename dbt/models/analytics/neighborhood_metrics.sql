{{ config(alias="neighborhood_metrics") }}
-- Per-barrio aggregates from spatial neighborhood_id (not scraped name).
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

FROM {{ source("core", "neighborhoods") }} AS n
LEFT JOIN {{ source("core", "listings") }} AS l
    ON l.neighborhood_id = n.id
    AND l.price_int > 0
    AND l.area_sqm > 0
GROUP BY n.id, n.name, n.geom
