{{ config(alias="neighborhood_tourism") }}

SELECT
    n.id AS neighborhood_id,
    n.name AS neighborhood_name,
    COUNT(ta.id)::BIGINT AS tourist_apt_count,
    ROUND(
        (COUNT(ta.id)::NUMERIC / NULLIF(nm.total_count, 0)) * 100,
        1
    ) AS tourist_density_pct
FROM {{ source("core", "neighborhoods") }} AS n
LEFT JOIN {{ source("core", "tourist_apartments") }} AS ta ON ta.neighborhood_id = n.id
LEFT JOIN {{ ref("neighborhood_metrics") }} AS nm ON nm.neighborhood_id = n.id
WHERE n.geom IS NOT NULL
GROUP BY n.id, n.name, nm.total_count
