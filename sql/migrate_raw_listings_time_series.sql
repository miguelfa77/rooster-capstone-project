-- Allow multiple observations per listing over time in raw.listings_raw
-- (was: PRIMARY KEY (url) + ON CONFLICT DO NOTHING → re-scrapes never inserted).
-- Run: psql -d rooster -f sql/migrate_raw_listings_time_series.sql

-- Normalize null scraped_at for PK (should be rare)
UPDATE raw.listings_raw SET scraped_at = '' WHERE scraped_at IS NULL;

ALTER TABLE raw.listings_raw DROP CONSTRAINT IF EXISTS listings_raw_pkey;

-- In case an old load created duplicate (url, scraped_at), keep one row
DELETE FROM raw.listings_raw a
USING raw.listings_raw b
WHERE a.ctid < b.ctid
  AND a.url = b.url
  AND a.scraped_at IS NOT DISTINCT FROM b.scraped_at;

ALTER TABLE raw.listings_raw ADD PRIMARY KEY (url, scraped_at);
