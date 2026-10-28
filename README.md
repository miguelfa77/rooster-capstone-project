# Rooster

**Rooster** is a Valencia-focused real estate intelligence app: a **Streamlit** UI with a chat copilot (OpenAI function calling + PostgreSQL/PostGIS) and an **Inteligencia** tab for maps and market metrics.

**Live app:** [https://rooster-capstone-project-production.up.railway.app](https://rooster-capstone-project-production.up.railway.app)

---

## Repository layout

| Path | Purpose |
|------|---------|
| **`app.py`** | Streamlit entrypoint: chat tab, Intel tab, caching, Folium/Plotly. |
| **`agent/`** | LLM agent (`agent_pipeline.py`), tool schemas (`openai_tools.py`), UI strings in Spanish (`ui_es.py`), renderers (`renderers.py`), DB helpers + static schema text (`llm_sql.py`). |
| **`api/`** | FastAPI service exposing the agent: conversations, streamed turns (SSE), Intel data. See **API** below. |
| **`tests/`** | `unit/` and `api/` (pytest), `eval/` (agent baseline eval against a live model). |
| **`pipeline/`** | Data loaders and scrapers: `raw/` (CSV → Postgres), `core/` (enrichment), `idealista/`, `open_data/` (transit, tourist apartments). |
| **`sql/`** | SQL migrations: bootstrap, `core` DDL, enrichment, open-data tables — not the `analytics` views (those are built by **dbt**). |
| **`dbt/`** | dbt project: `analytics.*` views + `dbt test`. |
| **`scripts/`** | `sync_local_to_railway.sh` — pg_dump local → wipe Railway DB → restore (see `scripts/.env.example`). |
| **`interface/`** | Static assets (e.g. icons). |
| **`requirements.txt`** | Python dependencies (Streamlit app, pipelines, scrapers, and **dbt**). |
| **`Procfile`** | Process type for Railway: `streamlit run app.py` on `$PORT`. |

This matches a common small **data + app** repo: **application code** at the root or in `agent/`, **ETL** under `pipeline/`, **DDL** under `sql/`, and **infra hints** via `Procfile` and env-based configuration.

---

## Configuration

- **`DATABASE_URL`** or **`PGHOST`** / **`PGPORT`** / **`PGUSER`** / **`PGPASSWORD`** / **`PGDATABASE`** — same database the loaders and app use. The app and `agent/llm_sql.py` also load **`agent/.env`**; pipeline CLIs load **`pipeline/.env`** and merge unset keys.
- **`OPENAI_API_KEY`** (or **`OPENAI_KEY`**) — required for chat.
- **`ROOSTER_API_KEY`** — shared secret for the API (`X-Rooster-Key` header). If unset, the API rejects every request.

Optional open-data (transit + tourist apartments): set **`TOURIST_APT_CSV_URL`** or **`TOURIST_APT_CSV_PATH`**, **`TOURIST_APT_PROVINCE_CODES`** (default `46` for Valencia province), and related vars as documented in **`pipeline/.env.example`** when you load VUT data.

---

## SQL apply order (Postgres)

Run scripts against your `rooster` (or equivalent) database in dependency order, for example:

1. `sql/bootstrap_rooster.sql`
2. `sql/core_tables.sql`
3. Migrations as needed (`sql/migrate_*.sql`; the API needs `sql/migrate_app_conversations.sql`, which creates the `app` schema)
4. Spatial / enrichment SQL as needed (e.g. `sql/match_listings_neighborhood_spatial.sql`, `sql/enrich_listings_refresh.sql`)
5. `sql/open_data_tables.sql` (transit + tourist apartments DDL and `nearest_stop_m` refresh)
6. **`dbt run`** from `dbt/` — builds all **`analytics.*`** views (see the **dbt** section below). Run `dbt test` optionally.

PostGIS must be enabled where you use spatial columns. Railway’s default Postgres plugin may need PostGIS installed separately for spatial features.

---

## dbt (analytics views and tests)

The **`dbt/`** project defines **`analytics`** views (`neighborhood_metrics`, `listing_summary`, `price_changes`, `neighborhood_transport`, `neighborhood_tourism`, `neighborhood_profile`) and runs [tests](https://docs.getdbt.com/docs/build/tests) (`dbt test`) against them.

**Prerequisites:** Postgres + PostGIS; `sql/bootstrap_rooster.sql`, `sql/core_tables.sql`, your usual loaders, and **`sql/open_data_tables.sql`** so `core.transit_stops` and `core.tourist_apartments` exist — dbt does not create those tables.

**First time in `dbt/`:**

```bash
cp profiles.yml.example profiles.yml
```

Copy **`dbt/.env.example`** to **`dbt/.env`** (gitignored) with `PGHOST`, `PGPORT`, `PGUSER`, `PGPASSWORD`, `PGDATABASE`, and **`DBT_PROFILES_DIR`** (e.g. `.` when your shell’s working directory is `dbt/`, or an absolute path to `dbt/`).

**Run** (from the **`dbt/`** directory so `DBT_PROFILES_DIR=.` resolves correctly if you use it):

```bash
pip install -r requirements.txt
cd dbt
set -a && source .env && set +a
dbt run
dbt test
```

dbt does not read `.env` files by itself; `source` exports variables for `profiles.yml`. If credentials live only in **`agent/.env`**, use `set -a && source ../agent/.env && source .env && set +a` instead.

The Streamlit app does not invoke dbt on Railway; run dbt locally or in CI when you change models or want regression checks.

---

## API

The agent runs behind a FastAPI service (`api/`) so other clients (a web app, a voice line) can use it. Streamlit and the API share the same turn logic (`agent/turn.py`) and Intel queries (`agent/intel_data.py`).

**Run locally** (same database and env vars as the app, plus `ROOSTER_API_KEY`; apply `sql/migrate_app_conversations.sql` once):

```bash
pip install -r api/requirements.txt
uvicorn api.main:app --port 8000
```

**Docker** (build from the repo root): `docker build -f api/Dockerfile -t rooster-api .`

**Endpoints** — all except `/health` need the `X-Rooster-Key` header; the API is meant to be called by trusted servers, never directly from a browser.

| Method | Path | Returns |
|--------|------|---------|
| `GET` | `/health` | `{"ok": true}` |
| `POST` | `/conversations` | `201 {"id"}` |
| `POST` | `/conversations/{id}/messages` | Body `{"text"}` (1–2000 chars). Streams Server-Sent Events. `404` unknown id, `409` a turn is already running. |
| `GET` | `/conversations/{id}` | `{"id", "messages"}` |
| `GET` | `/intel/framing`, `/intel/neighborhoods?metric=…`, `/intel/overlays/transit`, `/intel/overlays/tourism`, `/intel/listings?operation=All\|venta\|alquiler`, `/intel/parcels`, `/intel/ranking`, `/intel/yield-liquidity` | Intel tab data (cached 5–10 min) |

**SSE events** for one turn, in order: `thought` (progress text), `text_delta` (answer text), `primitive` (one chart/table/KPI with its own `data`), `follow_ups`, then `done`. A failure sends `error` (`reason`, `suggestion`) then `done`. `done` is sent only after the turn is saved. Event and primitive schemas are in `api/openapi.json` (regenerate with `python -m api.export_openapi`).

**Tests**

```bash
pytest                                   # unit + API tests
TEST_DATABASE_URL=postgresql+psycopg2://... pytest tests/api/test_store.py   # store tests need a disposable Postgres (they drop the app schema)
python -m tests.eval.run_eval compare    # agent eval vs tests/eval/baseline.json (live model + data)
python -m tests.eval.run_eval compare --via-api http://localhost:8000
```

---

## Deploying (Railway)

- Connect the GitHub repo; Railway runs the **`Procfile`** web process.
- Add a **PostgreSQL** plugin and set **`DATABASE_URL`** on the web service (or equivalent `PG*` variables).
- Set **`OPENAI_API_KEY`** on the web service.
- Schema and data are **not** applied automatically: run `sql/*.sql`, **`dbt run`** for `analytics` views, or copy your DB to Railway with **`bash scripts/sync_local_to_railway.sh`** (configure `scripts/.env` from `scripts/.env.example`).

---

## Local development

If you run from source instead of the hosted app:

```bash
./bin/python -m streamlit run app.py
```

Use a **local** project path (not iCloud-synced Desktop/Documents) if Streamlit hits file open timeouts.

---

## Troubleshooting

### `TimeoutError` in Streamlit / `tokenize.open`

If the project lives on a **cloud-synced or slow volume**, move the repo to a local folder and restart Streamlit.
