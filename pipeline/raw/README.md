# Raw loaders

Load raw data into PostgreSQL (`raw.*` schema). Run from the **repo root**.

## Prerequisites

1. PostgreSQL with PostGIS:
   ```bash
   createdb rooster
   psql -d rooster -f sql/bootstrap_rooster.sql
   ```

2. Config: `pipeline/.env` with PGHOST, PGPORT, PGUSER, PGPASSWORD, PGDATABASE

3. Dependencies: `pip install -r agent/requirements.txt` (psycopg2-binary, pandas, geopandas, pyshp, shapely)

## Load order

Loaders can run in any order (no dependencies between them):

```bash
python -m pipeline.raw.load_idealista_raw
python -m pipeline.raw.load_catastro_vias_raw
python -m pipeline.raw.load_barrios_raw
python -m pipeline.raw.load_catastro_parcels_raw
```

Or run all at once:

```bash
python -m pipeline.raw.run_all
```

## Data sources

| Loader | Source |
|--------|--------|
| load_idealista_raw | `pipeline/idealista/data/idealista_alquiler.csv`, `idealista_venta.csv` |
| load_catastro_vias_raw | `pipeline/catastro/data/catastro_vias.csv` or `pipeline/catastro/vias.csv` |
| load_barrios_raw | `pipeline/barrios/data/barris-barrios.geojson` |
| load_catastro_parcels_raw | `pipeline/catastro/data/46_UA_23012026_SHP.zip` |
