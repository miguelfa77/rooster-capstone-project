# Rooster

**Rooster** is a Valencia-focused real estate intelligence app: a Spanish chat copilot (OpenAI function calling + PostgreSQL/PostGIS) for investors, with maps and market metrics. Access is invite-only: people sign in with Google and only emails on an allowlist get in.

It runs as three services: a **Next.js** web app (`web/`) that talks only to a **FastAPI** service (`api/`), which runs the Python agent (`agent/`) and calls a **sandbox** service (`sandbox/`) for free-form chart code. All data lives in one PostGIS database.

---

## Repository layout

| Path | Purpose |
|------|---------|
| **`web/`** | Next.js (App Router) + TypeScript + Tailwind web app: Google sign-in (Auth.js), the signed-in shell, conversation history and the chat (sidebar, full screen at `/chat/<id>`, phone sheet). A same-origin proxy (`/api/rooster/*`) adds the API key and the signed-in user's id server-side; the browser never calls FastAPI directly. |
| **`api/`** | FastAPI service: users, user-owned conversations, streamed turns (SSE), Intel data. Publishes `api/openapi.json`. |
| **`agent/`** | The Python agent: planner/executor loop (`agent_pipeline.py`), synthesizer, one chat turn (`turn.py`), self-contained primitive payloads for clients (`primitive_payload.py`), Intel queries (`intel_data.py`), semantic layer, Spanish strings (`ui_es.py`). |
| **`sandbox/`** | FastAPI service that executes `code` primitives (Plotly / Folium) in pre-warmed worker processes. |
| **`pipeline/`** | Data loaders and scrapers: `raw/` (CSV → Postgres), `core/` (listings, streets, neighborhoods, parcels, enrichment), `idealista/`, `open_data/` (transit, tourist apartments). |
| **`dbt/`** | dbt project: `analytics.*` views + `dbt test`. The app only queries `analytics`. |
| **`sql/`** | SQL migrations: bootstrap, `core` DDL, enrichment, open-data tables, the API's `app` schema — not the `analytics` views (those are built by dbt). |
| **`tests/`** | `unit/` and `api/` (pytest), `eval/` (agent eval against a live model), `fixtures/` (shared primitive payloads, also used by the web tests). |
| **`scripts/`** | `sync_local_to_railway.sh` — pg_dump local → wipe Railway DB → restore (see `scripts/.env.example`). |
| **`docker-compose.yml`** | The whole stack locally: PostGIS, sandbox, API, web. |
| **`requirements.txt`** | Python dependencies for local development (agent, pipelines, scrapers, dbt, tests). `api/` and `sandbox/` have their own. |

---

## Configuration

Environment variables per service (copy **`.env.example`** to **`.env`**; it is gitignored):

| Service | Variables |
|---------|-----------|
| **db** (compose) | `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB` |
| **api** | `DATABASE_URL` (or `PGHOST` / `PGPORT` / `PGUSER` / `PGPASSWORD` / `PGDATABASE`), `OPENAI_API_KEY` (or `OPENAI_KEY`), `ROOSTER_API_KEY` (shared secret, `X-Rooster-Key` header; if unset the API rejects every request), `ROOSTER_SANDBOX_URL` (must include `http://`; default `http://localhost:8001`), `ROOSTER_MAX_TURNS_PER_MINUTE` (turns that may start per UTC minute across all conversations and API instances, counted in `app.turn_starts`; default `30`, `0` or negative disables, an invalid value falls back to `30` with a warning; over the cap → `429`) |
| **sandbox** | `PORT` (default `8080` in its Docker image) |
| **web** | All server-only: never prefix them with `NEXT_PUBLIC_`. `ROOSTER_API_URL` (e.g. `http://localhost:8000`), `ROOSTER_API_KEY` (same value as the API), `AUTH_SECRET` (signs the session cookie; generate with `openssl rand -base64 33`), `AUTH_URL` (the public origin of the web app, e.g. `http://localhost:3000`; Auth.js builds the Google callback URL from it, so it must match the redirect URI you registered. Compose defaults it to `http://localhost:3000`), `AUTH_GOOGLE_ID` and `AUTH_GOOGLE_SECRET` (the Google OAuth client, see **Google sign-in**), `ROOSTER_ALLOWED_EMAILS` (comma-separated Google emails allowed in, case-insensitive; empty means nobody can sign in). |
| **pipeline / dbt** | Same database as the API (`DATABASE_URL` or `PG*`). `agent/llm_sql.py` also loads **`agent/.env`**; pipeline CLIs load **`pipeline/.env`** and merge unset keys; dbt reads **`dbt/.env`** (see below). |

The turn limit is counted in the database (fixed one-minute windows), so it holds across any number of API instances; across a minute boundary up to twice the cap can start.

Optional agent tuning: `ROOSTER_PLANNER_MODEL`, `ROOSTER_SYNTHESIZER_MODEL`, `ROOSTER_MEMORY_MODEL` and the `ROOSTER_REASONING_*` variables (see `agent/config.py`).

Optional open-data (transit + tourist apartments): set **`TOURIST_APT_CSV_URL`** or **`TOURIST_APT_CSV_PATH`**, **`TOURIST_APT_PROVINCE_CODES`** (default `46` for Valencia province), and related vars read by `pipeline/open_data/` when you load VUT data.

---

## Running locally

### With Docker Compose

```bash
cp .env.example .env          # set OPENAI_API_KEY, ROOSTER_API_KEY, AUTH_SECRET, AUTH_GOOGLE_ID/SECRET, ROOSTER_ALLOWED_EMAILS
docker compose up --build
```

Web: <http://localhost:3000> · API: <http://localhost:8000> · Postgres: `localhost:5432` (credentials from `.env`). The database volume starts empty; load data once as described in **First run: data**.

### Without Docker

You need Postgres + PostGIS, Python 3.11 and Node 22.

```bash
pip install -r requirements.txt -r api/requirements.txt
set -a && source .env && set +a     # DATABASE_URL, OPENAI_API_KEY, ROOSTER_API_KEY, ROOSTER_SANDBOX_URL

# sandbox (own dependencies; port 8001 matches the default ROOSTER_SANDBOX_URL)
pip install -r sandbox/requirements.txt
(cd sandbox && uvicorn main:app --port 8001)

# API
uvicorn api.main:app --port 8000

# web (web/.env.local holds ROOSTER_API_URL, ROOSTER_API_KEY and the AUTH_* / allowlist variables; see web/.env.example)
cd web && npm ci && npm run dev     # http://localhost:3000
```

Run each long-running command in its own terminal.

### Google sign-in

Sign-in is Google only and invite-only. Create the OAuth client once:

1. In the [Google Cloud console](https://console.cloud.google.com/apis/credentials), create an **OAuth client ID** of type **Web application** (configure the consent screen first if asked).
2. Add the authorized redirect URI **`<AUTH_URL>/api/auth/callback/google`**, e.g. `http://localhost:3000/api/auth/callback/google` locally and `https://<your-domain>/api/auth/callback/google` when deployed (add one per origin you use).
3. Put the client id and secret in `AUTH_GOOGLE_ID` and `AUTH_GOOGLE_SECRET`, set `AUTH_SECRET`, and list the people you invite in `ROOSTER_ALLOWED_EMAILS`.

The allowlist is checked at sign-in and again on every request, so removing an email signs that person out on their next request. Someone not on the list lands on an "invitation only" page. The automated tests mint session cookies instead of talking to Google, so **the real Google round trip (consent screen, callback, allowlist, first-time user creation) must be tested once by hand with real credentials** before relying on it.

### First run: data

The app reads `analytics.*` views and the API stores users and conversations in the `app` schema. On a fresh database:

1. Apply the SQL in **SQL apply order** below (includes `sql/migrate_app_conversations.sql` then `sql/migrate_app_users.sql` for the API).
2. Load data: `python -m pipeline.raw.load_barrios_raw`, `python -m pipeline.core.run_all`, then the open-data loaders in `pipeline/open_data/` (transit, tourist apartments) — or restore an existing dump.
3. Build the analytics views: `cd dbt && dbt run` (see **dbt** below).

With compose, run these from the host against `localhost:5432` using the `DATABASE_URL` in `.env`.

**Upgrading from v3.** `sql/migrate_app_users.sql` adds users and makes every conversation belong to one. v3 conversations have no owner, so the migration **deletes them** (it is idempotent; run it once after `migrate_app_conversations.sql`). Back up `app.conversations` first if you need them.

**Conversations from earlier test runs.** Conversations stored by Phase 1 API test runs, before assistant messages carried the ordered `blocks` list, may fail to load in the web app. They hold no real data; clear them with `TRUNCATE app.conversations CASCADE;` (this also empties `app.messages`).

---

## SQL apply order (Postgres)

Run scripts against your `rooster` (or equivalent) database in dependency order, for example:

1. `sql/bootstrap_rooster.sql`
2. `sql/core_tables.sql`
3. Migrations as needed (`sql/migrate_*.sql`; the API needs `sql/migrate_app_conversations.sql`, which creates the `app` schema, then `sql/migrate_app_users.sql`, which adds users and makes conversations user-owned — it deletes existing conversations that have no owner)
4. Spatial / enrichment SQL as needed (e.g. `sql/match_listings_neighborhood_spatial.sql`, `sql/enrich_listings_refresh.sql`)
5. `sql/open_data_tables.sql` (transit + tourist apartments DDL and `nearest_stop_m` refresh)
6. **`dbt run`** from `dbt/` — builds all **`analytics.*`** views (see the **dbt** section below). Run `dbt test` optionally.

PostGIS must be enabled where you use spatial columns (the compose `db` image has it).

---

## dbt (analytics views and tests)

The **`dbt/`** project defines **`analytics`** views (`neighborhood_metrics`, `listing_summary`, `price_changes`, `neighborhood_transport`, `neighborhood_tourism`, `neighborhood_profile`) and runs [tests](https://docs.getdbt.com/docs/build/tests) (`dbt test`) against them.

**Prerequisites:** Postgres + PostGIS; `sql/bootstrap_rooster.sql`, `sql/core_tables.sql`, your usual loaders, and **`sql/open_data_tables.sql`** so `core.transit_stops` and `core.tourist_apartments` exist — dbt does not create those tables.

**First time in `dbt/`:**

```bash
cp profiles.yml.example profiles.yml
```

Create **`dbt/.env`** (gitignored) with `PGHOST`, `PGPORT`, `PGUSER`, `PGPASSWORD`, `PGDATABASE`, and **`DBT_PROFILES_DIR`** (e.g. `.` when your shell’s working directory is `dbt/`, or an absolute path to `dbt/`).

**Run** (from the **`dbt/`** directory so `DBT_PROFILES_DIR=.` resolves correctly if you use it):

```bash
pip install -r requirements.txt
cd dbt
set -a && source .env && set +a
dbt run
dbt test
```

dbt does not read `.env` files by itself; `source` exports variables for `profiles.yml`. If credentials live only in **`agent/.env`**, use `set -a && source ../agent/.env && source .env && set +a` instead.

No service runs dbt for you; run it locally or in CI when you change models or want regression checks.

---

## API

The agent runs behind a FastAPI service (`api/`); the web app is its client. Turn logic lives in `agent/turn.py` and Intel queries in `agent/intel_data.py`.

**Docker** (build from the repo root): `docker build -f api/Dockerfile -t rooster-api .`

**Endpoints** — all except `/health` need the `X-Rooster-Key` header, and everything user-scoped (`/me`, `/conversations…`) also needs `X-Rooster-User: <user id>`. The API trusts the web server about who the user is because only the web server knows the key; it is meant to be called by trusted servers (the web app's proxy), never directly from a browser. Another user's conversation answers `404`.

| Method | Path | Returns |
|--------|------|---------|
| `GET` | `/health` | `{"ok": true}` |
| `PUT` | `/users` | Body `{"email", "name", "image"}`. Creates or updates the user (called by the web server at sign-in); returns the user with its `id`. Needs only the key. |
| `GET` | `/me` | The user: `id`, `email`, `name`, `image`, `preferences` |
| `PATCH` | `/me/preferences` | Partial update of the user's stored preferences (theme, chat sidebar open) |
| `GET` | `/conversations` | The user's conversations, newest first: `id`, `title`, `updated_at` |
| `POST` | `/conversations` | `201 {"id"}` |
| `PATCH` | `/conversations/{id}` | Body `{"title"}` (1–120 chars). Renames. |
| `DELETE` | `/conversations/{id}` | `204` |
| `POST` | `/conversations/{id}/messages` | Body `{"text"}` (1–2000 chars). Streams Server-Sent Events. `404` unknown id, `409` a turn is already running, `429` the turn limit was reached. |
| `GET` | `/conversations/{id}` | `{"id", "title", "messages"}`; assistant messages carry the ordered `blocks` the web renders |
| `GET` | `/intel/framing`, `/intel/neighborhoods?metric=…`, `/intel/overlays/transit`, `/intel/overlays/tourism`, `/intel/listings?operation=All\|venta\|alquiler`, `/intel/parcels`, `/intel/ranking`, `/intel/yield-liquidity` | Intel page data (cached 5–10 min) |

**SSE events** for one turn, in order: `thought` (progress text), `text_delta` (answer text), `primitive` (one chart/table/KPI with its own `data`), `follow_ups`, then `done`. A failure sends `error` (`reason`, `suggestion`) then `done`. `done` is sent only after the turn is saved.

**Regenerating types.** Event, primitive, message and Intel schemas are in `api/openapi.json`; the web's `web/src/lib/api-types.ts` is generated from it. After changing `api/models.py` or the events:

```bash
python3 -m api.export_openapi && cd web && npm run gen:types
```

`npm run check:types` fails when the generated file is stale.

---

## What v4 Phase 1 includes, and what's next

**Included:** invite-only Google sign-in, users and per-user conversations in the API, a rebuilt web app (light/dark theme remembered per user, left navigation, conversation history with rename and delete, the chat as a sidebar, a full-screen page and a phone sheet, compact answer cards that open full-size), and end-to-end tests. The agent itself did not change.

**Not yet:** the other navigation sections (Mapa, Barrios, Anuncios, Comparar, Calculadora, Seguimiento) show as "Pronto", the old Inteligencia page is no longer in the app (its API endpoints remain), and there is no shared-conversation link yet. See `docs/superpowers/specs/2026-11-12-v4-product-design.md` for the plan.

---

## Tests

```bash
python3 -m pytest -q                     # Python unit + API tests
TEST_DATABASE_URL=postgresql+psycopg2://... python3 -m pytest tests/api/test_store.py   # store tests need a disposable Postgres (they drop the app schema)

cd web
npm test                                 # Vitest unit tests
npm run typecheck && npm run check:types
npm run e2e                              # Playwright against a fake API (web/tests/fake-api); builds and starts Next on :3100, signed in with a minted session cookie
```

The end-to-end tests need a Chromium for Playwright (`npx playwright install chromium`). If that download is not possible, point them at an installed Chrome/Chromium with `PLAYWRIGHT_CHROMIUM_EXECUTABLE=/path/to/chrome npm run e2e`.

### Agent eval (live model and data)

The eval runs fixed Spanish questions (`tests/eval/questions.yml`) and compares the tools called and primitive kinds returned; answer text is printed for review, never diffed. It needs `DATABASE_URL` and `OPENAI_API_KEY` against a loaded database. Run it once after setting up the stack:

```bash
python3 -m tests.eval.run_eval record                                   # baseline, run directly through agent.turn
ROOSTER_API_KEY=... python3 -m tests.eval.run_eval compare --via-api http://localhost:8000   # same questions through the API
```

`record` writes `tests/eval/baseline.json`; `compare --via-api` reruns the questions through the running API and diffs against it.

**Known gap:** no baseline was ever recorded for the old Streamlit (v2) app, so this checks the API against the direct turn, not against v2 behavior.

---

## Deploying

No hosting configuration is checked in. Each service has its own Dockerfile: `api/Dockerfile` (build from the repo root), `sandbox/Dockerfile` (build from `sandbox/`), `web/Dockerfile` (build from `web/`). Set the variables in **Configuration** on each service; `ROOSTER_SANDBOX_URL` and `ROOSTER_API_URL` must point at the sandbox and API over the host's private network.
The sandbox runs model-written code: keep it private (reachable only by the API) and without internet egress.

Schema and data are **not** applied automatically: run `sql/*.sql` and **`dbt run`**, or copy a local database to Railway with **`bash scripts/sync_local_to_railway.sh`** (configure `scripts/.env` from `scripts/.env.example`).
