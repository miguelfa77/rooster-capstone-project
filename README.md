# rooster-capstone-project
Rooster - Your AI Data Intern for Real Estate Investors in Valencia

The Streamlit UI is in **Spanish** (`agent/ui_es.py`).

## Open data (transit + tourist apartments)

Apply SQL in order (see [`sql/README.md`](sql/README.md)): after `analytics_views.sql`, run **`open_data_tables.sql`** then **`open_data_views.sql`**.

**Environment**

- **`DATABASE_URL`** or **`PG*`** – Same DB as the app. Pipeline CLIs load **`agent/.env`**, **`pipeline/.env`**, and **repo root `.env`** (last two merge keys that are unset or empty for `PG*` / `DATABASE_URL`). A lone `PGPASSWORD=…` is enough if you use defaults `localhost` / `postgres` / `rooster`.
- **`TOURIST_APT_CSV_URL`** – GVA CSV (e.g. [lista VUT](https://dadesobertes.gva.es/)); see **`pipeline/.env.example`**. If unset or the download fails, use **`TOURIST_APT_CSV_PATH`**.
- **`TOURIST_APT_CSV_PATH`** – Local CSV fallback.
- **`TOURIST_APT_PROVINCE_CODES`** – INE province codes to **include** (default **`46`** = provincia de Valencia, all municipios). Example: `46` only, or `46,12` for Valencia + Castelló.
- **`TOURIST_APT_GEOCODE_UNMATCHED`** – Set to `1` to geocode addresses without a **`ref_catastral`** match in **`core.parcels`** via Nominatim (slow). By default only parcel-centroid joins are used; **`core.parcels`** is often **Valencia city** only — VUTs elsewhere in the province will not get coordinates until parcels cover those municipios or you enable geocoding.

**Commands**

```bash
python -m pipeline.open_data.fetch_transit_overpass
python -m pipeline.open_data.load_tourist_apartments
```

Re-run **`fetch_transit_overpass`** after changing stops data; **`open_data_tables.sql`** defines `nearest_stop_m` and the loaders refresh it.

After **`load_tourist_apartments`**, optional street geocoding (fuzzy match to **`core.streets`** + **`core.parcels`**, Valencia city): `python -m pipeline.open_data.geocode_tourist_apartments` (see `sql/migrate_tourist_apartments_geocode.sql`).

## Troubleshooting

### `TimeoutError: [Errno 60] Operation timed out` in `tokenize.open` / Streamlit `get_bytecode`

Streamlit reads `app.py` from disk to compile/cache it. A **timeout while opening the file** almost always means the path is on a **slow or cloud-synced volume** (common: **Desktop** or **Documents** with **iCloud Drive**, Dropbox, Google Drive, NFS, or an external disk that sleeps).

**What to do**

1. Move the repo to a **fully local** folder, e.g. `~/dev/rooster-capstone-project` or `~/Projects/rooster-capstone-project`, and run Streamlit from there.
2. Or turn off iCloud sync for Desktop & Documents (Apple ID → iCloud → iCloud Drive → Options), or keep the project outside synced folders.
3. After moving, restart Streamlit from the new path.

This is an environment/filesystem issue, not a bug in `app.py` itself.
