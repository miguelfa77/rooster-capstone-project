{{ config(alias="listing_summary") }}

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
FROM {{ source("core", "listings") }}
GROUP BY operation
