# Rooster SQL

Run in order:

1. **bootstrap_rooster.sql** – PostGIS, schemas (raw, core, analytics), raw tables
2. **core_tables.sql** – Core tables (includes `core.listing_snapshots` and listing history columns)
3. **migrate_listing_upsert_history.sql** – Only if you already had a DB before these columns existed (adds `first_seen_at`, `last_seen_at`, `price_int_previous`, `listing_snapshots`)
4. **migrate_raw_listings_time_series.sql** – If your DB still has `raw.listings_raw` with `PRIMARY KEY (url)` only, run this so repeated scrapes (same URL, new `scraped_at`) can load into raw and `core.listing_snapshots`.
5. **migrate_listing_enrichment.sql** – Adds amenity booleans, `floor_int`, `minutes_to_center`, `lat`/`lng`, `geocode_quality`, generated `days_on_market`, pg_trgm indexes; runs initial regex updates.
6. **enrich_geocode.sql** (optional) – After streets/parcels/listings are loaded: cleans `street_name_raw`, resets prior neighborhood-only pins, normalizes avenue/calle forms, picks best `core.streets` row by similarity (`> 0.3`), joins parcels with **`core.parcels.via = core.streets.codigo_via`**, neighborhood fallback, then a summary `SELECT` on `geocode_quality`. Requires `pg_trgm` + `unaccent`. Or: `GEOCODE_LISTINGS=1 python -m pipeline.core.enrich_listings`.
7. **match_listings_neighborhood_spatial.sql** (optional) – After geocoding: sets **`core.listings.neighborhood_id`** where `ST_Within(point, core.neighborhoods.geom)` (WGS84). Prints coverage counts. Re-run when new listings get coordinates.
8. **analytics_views.sql** – `neighborhood_metrics` (incl. `avg_days_on_market`), `listing_summary`, `price_changes`. Re-run after schema changes.
9. **open_data_tables.sql** – Raw/core transit + tourist tables, `core.listings.nearest_stop_m`, indexes. Run after PostGIS + neighborhoods + listings exist.
10. **open_data_views.sql** – `analytics.neighborhood_transport`, `neighborhood_tourism`, `neighborhood_profile`. **Must run after** `analytics_views.sql` (depends on `neighborhood_metrics`).
11. **migrate_tourist_apartments_geocode.sql** (optional) – After `load_tourist_apartments`: parse addresses, fuzzy-match `core.streets`, join `core.parcels`, assign `core.neighborhoods`, refresh `core.tourist_apartments`. Or: `python -m pipeline.open_data.geocode_tourist_apartments` (same SQL + coverage report). Requires `pg_trgm` + `unaccent` + Valencia city streets/parcels.

Then run Python loaders:
- `python -m pipeline.raw.run_all`
- `python -m pipeline.core.run_all`

Open data (after steps 9–10; set `DATABASE_URL` or `PG*` as for other pipeline commands, e.g. `pipeline/.env`):

- `python -m pipeline.open_data.fetch_transit_overpass` – Overpass → `raw`/`core.transit_stops`, refreshes `nearest_stop_m` on listings.
- `python -m pipeline.open_data.load_tourist_apartments` – GVA tourist CSV; set `TOURIST_APT_CSV_URL` / `TOURIST_APT_PROVINCE_CODES` (see `pipeline/.env.example`). Coordinates from **`ref_catastral` → `core.parcels.refcat`** when it matches; then run **`python -m pipeline.open_data.geocode_tourist_apartments`** for street/parcel fallback (Valencia city streets/parcels only).
