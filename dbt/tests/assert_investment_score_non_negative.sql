SELECT neighborhood_id
FROM {{ ref("neighborhood_profile") }}
WHERE investment_score < 0
