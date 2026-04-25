{{ config(alias="parcel_metrics") }}

SELECT
    n.id::text AS neighborhood_id,
    n.name AS neighborhood_name,
    COUNT(p.refcat) AS parcel_count,
    MIN(p.area) AS min_parcel_area_sqm,
    MAX(p.area) AS max_parcel_area_sqm,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY p.area) AS median_parcel_area_sqm
FROM core.neighborhoods n
INNER JOIN core.parcels p
    ON ST_Intersects(
        n.geom,
        ST_Transform(p.geom, 4326)
    )
GROUP BY n.id, n.name
