-- Yield from medians should not be negative; extreme high values flag bad inputs.
SELECT neighborhood_id
FROM {{ ref("neighborhood_metrics") }}
WHERE gross_rental_yield_pct IS NOT NULL
  AND (gross_rental_yield_pct < 0 OR gross_rental_yield_pct > 500)
