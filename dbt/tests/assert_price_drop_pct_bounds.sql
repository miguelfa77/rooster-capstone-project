-- Percent drop vs previous ask should stay within [0, 100] for typical listings.
SELECT url
FROM {{ ref("price_changes") }}
WHERE price_drop_pct < 0
   OR price_drop_pct > 100
