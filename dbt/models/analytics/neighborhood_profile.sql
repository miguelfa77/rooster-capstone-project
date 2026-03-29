{{ config(alias="neighborhood_profile") }}

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
    COALESCE(nto.tourist_apt_count, 0::BIGINT) AS tourist_apt_count,
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
        )::NUMERIC,
        2
    ) AS investment_score
FROM {{ ref("neighborhood_metrics") }} AS nm
LEFT JOIN {{ ref("neighborhood_transport") }} AS nt ON nt.neighborhood_id = nm.neighborhood_id
LEFT JOIN {{ ref("neighborhood_tourism") }} AS nto ON nto.neighborhood_id = nm.neighborhood_id
