{{ config(alias="price_changes") }}

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
FROM {{ source("core", "listings") }}
WHERE price_int_previous IS NOT NULL
  AND price_int IS NOT NULL
  AND price_int < price_int_previous
