-- Add upsert / history columns and listing_snapshots (existing databases).
-- Run: psql -d rooster -f sql/migrate_listing_upsert_history.sql

ALTER TABLE core.listings ADD COLUMN IF NOT EXISTS first_seen_at TIMESTAMPTZ;
ALTER TABLE core.listings ADD COLUMN IF NOT EXISTS last_seen_at TIMESTAMPTZ;
ALTER TABLE core.listings ADD COLUMN IF NOT EXISTS price_int_previous INTEGER;

UPDATE core.listings
SET
    first_seen_at = COALESCE(first_seen_at, NOW()),
    last_seen_at = COALESCE(last_seen_at, NOW())
WHERE first_seen_at IS NULL OR last_seen_at IS NULL;

CREATE TABLE IF NOT EXISTS core.listing_snapshots (
    id         BIGSERIAL PRIMARY KEY,
    url        TEXT NOT NULL,
    price_int  INTEGER,
    scraped_at TIMESTAMPTZ NOT NULL,
    UNIQUE (url, scraped_at)
);
CREATE INDEX IF NOT EXISTS listing_snapshots_url_idx ON core.listing_snapshots (url);
