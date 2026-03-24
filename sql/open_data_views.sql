-- Open data analytics views. Run AFTER sql/analytics_views.sql
-- (requires analytics.neighborhood_metrics).

DROP VIEW IF EXISTS analytics.neighborhood_profile CASCADE;
DROP VIEW IF EXISTS analytics.neighborhood_tourism CASCADE;
DROP VIEW IF EXISTS analytics.neighborhood_transport CASCADE;

-- ---------------------------------------------------------------------------
-- Stops per barrio + avg distance (m) from barrio centroid to nearest 3 stops citywide
-- ---------------------------------------------------------------------------
CREATE VIEW analytics.neighborhood_transport AS
SELECT
    n.id AS neighborhood_id,
    n.name AS neighborhood_name,
    COUNT(t.osm_id)::bigint AS transit_stop_count,
    (
        SELECT AVG(s.d)
        FROM (
            SELECT ST_Distance(ST_Centroid(n.geom)::geography, t2.geom::geography) AS d
            FROM core.transit_stops t2
            ORDER BY ST_Centroid(n.geom) <-> t2.geom
            LIMIT 3
        ) s
    ) AS avg_dist_to_stop_m
FROM core.neighborhoods n
LEFT JOIN core.transit_stops t ON t.neighborhood_id = n.id
WHERE n.geom IS NOT NULL
GROUP BY n.id, n.name, n.geom;

-- ---------------------------------------------------------------------------
-- Tourist apartments per barrio + density vs Idealista listing count
-- ---------------------------------------------------------------------------
CREATE VIEW analytics.neighborhood_tourism AS
SELECT
    n.id AS neighborhood_id,
    n.name AS neighborhood_name,
    COUNT(ta.id)::bigint AS tourist_apt_count,
    ROUND(
        (COUNT(ta.id)::numeric / NULLIF(nm.total_count, 0)) * 100,
        1
    ) AS tourist_density_pct
FROM core.neighborhoods n
LEFT JOIN core.tourist_apartments ta ON ta.neighborhood_id = n.id
LEFT JOIN analytics.neighborhood_metrics nm ON nm.neighborhood_id = n.id
WHERE n.geom IS NOT NULL
GROUP BY n.id, n.name, nm.total_count;

-- ---------------------------------------------------------------------------
-- Master profile: listing metrics + transport + tourism + composite score
-- ---------------------------------------------------------------------------
CREATE VIEW analytics.neighborhood_profile AS
SELECT
    nm.neighborhood_id,
    nm.neighborhood_name,
    nm.geom,
    nm.alquiler_count,
    nm.avg_alquiler_price,
    nm.min_alquiler_price,
    nm.max_alquiler_price,
    nm.median_alquiler_price,
    nm.avg_alquiler_area_sqm,
    nm.avg_alquiler_rooms,
    nm.median_alquiler_eur_per_sqm,
    nm.venta_count,
    nm.avg_venta_price,
    nm.min_venta_price,
    nm.max_venta_price,
    nm.median_venta_price,
    nm.avg_venta_area_sqm,
    nm.avg_venta_rooms,
    nm.median_venta_eur_per_sqm,
    nm.total_count,
    nm.gross_rental_yield_pct,
    nm.avg_days_on_market,
    nt.transit_stop_count,
    nt.avg_dist_to_stop_m,
    CASE
        WHEN nt.avg_dist_to_stop_m IS NULL THEN 'Unknown'
        WHEN nt.avg_dist_to_stop_m < 300 THEN 'Excellent'
        WHEN nt.avg_dist_to_stop_m < 600 THEN 'Good'
        WHEN nt.avg_dist_to_stop_m < 1000 THEN 'Moderate'
        ELSE 'Poor'
    END AS transport_rating,
    COALESCE(nto.tourist_apt_count, 0::bigint) AS tourist_apt_count,
    nto.tourist_density_pct,
    CASE
        WHEN nto.tourist_density_pct IS NULL THEN 'Unknown'
        WHEN nto.tourist_density_pct > 20 THEN 'High'
        WHEN nto.tourist_density_pct > 10 THEN 'Moderate'
        ELSE 'Low'
    END AS tourism_pressure,
    ROUND(
        (
            COALESCE(nm.gross_rental_yield_pct, 0) * 0.5
            + CASE
                WHEN COALESCE(nt.avg_dist_to_stop_m, 99999) < 400 THEN 2.0
                WHEN COALESCE(nt.avg_dist_to_stop_m, 99999) < 700 THEN 1.0
                ELSE 0.0
            END
            + CASE WHEN COALESCE(nto.tourist_density_pct, 0) < 10 THEN 1.0 ELSE 0.0 END
        )::numeric,
        2
    ) AS investment_score
FROM analytics.neighborhood_metrics nm
LEFT JOIN analytics.neighborhood_transport nt ON nt.neighborhood_id = nm.neighborhood_id
LEFT JOIN analytics.neighborhood_tourism nto ON nto.neighborhood_id = nm.neighborhood_id;
