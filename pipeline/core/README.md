# Core loaders

Transform raw → core (cleaned, typed columns). Run from **repo root**.

**Prerequisites:** Raw tables must be loaded first (`python -m pipeline.raw.run_all`).

```bash
# 1. Create core tables (once)
psql -U postgres -h localhost -d rooster -f sql/core_tables.sql

# 2. Run core loaders
python -m pipeline.core.run_all
```

**What each loader does:**
- `load_listings` – Upserts `core.listings` by `url` (updates `last_seen_at`, `price_int_previous`, other fields); appends `core.listing_snapshots` per raw row. Optional: `LISTINGS_TRUNCATE_BEFORE_LOAD=1` for a full rebuild from `raw.listings_raw`. After pulling schema changes, run `sql/migrate_listing_upsert_history.sql` once if the DB already existed.
- `load_streets` – Valencia city streets from vías
- `load_neighborhoods` – Direct copy from raw
- `load_parcels` – Direct copy (deduplicated by refcat)
- `enrich_listings` – After the above, recomputes amenity flags from `description`, `floor_int`, `minutes_to_center` (see `sql/enrich_listings_refresh.sql`). Requires `sql/migrate_listing_enrichment.sql` (or a fresh `sql/core_tables.sql`) once. Optional parcel geocode: `GEOCODE_LISTINGS=1 python -m pipeline.core.enrich_listings` (or run `sql/enrich_geocode.sql`) to fill `lat`/`lng` using `pg_trgm` + `core.parcels`.
