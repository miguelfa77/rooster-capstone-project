{{ config(alias="neighborhood_transport") }}

SELECT
    n.id AS neighborhood_id,
    n.name AS neighborhood_name,
    COUNT(t.osm_id)::BIGINT AS transit_stop_count,
    (
        SELECT AVG(s.d)
        FROM (
            SELECT ST_Distance(ST_Centroid(n.geom)::geography, t2.geom::geography) AS d
            FROM {{ source("core", "transit_stops") }} AS t2
            ORDER BY ST_Centroid(n.geom) <-> t2.geom
            LIMIT 3
        ) s
    ) AS avg_dist_to_stop_m
FROM {{ source("core", "neighborhoods") }} AS n
LEFT JOIN {{ source("core", "transit_stops") }} AS t ON t.neighborhood_id = n.id
WHERE n.geom IS NOT NULL
GROUP BY n.id, n.name, n.geom
