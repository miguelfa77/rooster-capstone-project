# Deploy Rooster on Railway

## 1. Create the project

1. In [Railway](https://railway.app/), **New project** → **Deploy from GitHub repo** and select this repository.
2. Railway will detect **Python** (via root `requirements.txt`) and use the `Procfile` `web` process to run Streamlit.

## 2. Add PostgreSQL

1. In the project, **Add** → **Database** → **PostgreSQL**.
2. Open your **web service** (the Streamlit app) → **Variables**.
3. Add a variable that points at the plugin DB, e.g.:
   - **Name:** `DATABASE_URL`
   - **Value:** `${{Postgres.DATABASE_URL}}`  
   (Use the exact reference name Railway shows for your Postgres service if it differs.)

Alternatively, **link** the Postgres service to the web service so `DATABASE_URL` is injected automatically (Railway UI may offer this when both services exist).

## 3. Required environment variables

Set these on the **web** service (Variables tab):

| Variable | Notes |
|----------|--------|
| `DATABASE_URL` | From Postgres (see above). Railway URLs usually work with `psycopg2` as-is; if SSL errors appear, try appending `?sslmode=require`. |
| `OPENAI_API_KEY` | Your OpenAI API key (same as local). Optional alias: `OPENAI_KEY` is mapped in code. |

Optional (if you use them in the app):

- `GOOGLE_API_KEY` / Gemini-related vars if used
- Any other keys your `README` lists for open data / tourist apartments

**Do not** commit secrets; set them only in Railway.

## 4. Schema and data

Railway only runs the app; it does **not** run your SQL migrations automatically. Load schema and data the same way you do locally:

- Run `sql/*.sql` against the Railway Postgres (one-off from your machine using `DATABASE_URL`, or a temporary job).
- Or use Railway’s **one-off shell** / **Postgres** connect string with `psql`.

### Restore a `pg_dump` from your laptop (paste variables, no URL)

Passwords with `@ : / # %` break `postgresql://…` URLs. Use the helper script and **discrete** `PG*` vars:

1. `cp scripts/railway-restore.env.example scripts/railway-restore.env`
2. Railway → **Postgres** service → **Variables** — copy **PGHOST** (public proxy host), **PGPORT**, **PGUSER**, **PGPASSWORD**, **PGDATABASE** into `scripts/railway-restore.env` (gitignored).
3. Place your `pg_dump -Fc` file at the repo root as `rooster.dump`, or set `DUMP_FILE=` in that file.
4. `bash scripts/railway_pg_restore.sh test` then `bash scripts/railway_pg_restore.sh`

If **`password authentication failed`**: Prefer pasting **`DATABASE_URL`** from the **Postgres** service (Variables / Connect) into `scripts/railway-restore.env` — it matches what Railway expects. If you only set **`POSTGRES_PASSWORD`** in the UI but the data directory was created earlier, the **live password inside Postgres can still be the old one** until you **regenerate credentials**, **reset the DB password** in Railway, or **wipe the volume** and let Postgres init again. Use **`PGSSLMODE=disable`** (or **`DATABASE_URL`** with `sslmode=disable`) for the public TCP proxy unless Railway gives you a TLS URL.

Use a **PostGIS-enabled** Postgres for spatial dumps; the default Railway Postgres plugin may not include PostGIS.

## 5. Deploy and open the URL

After deploy succeeds, Railway assigns a **public URL** (Settings → Networking / Generate domain). Open it in the browser.

## 6. Troubleshooting

- **`ModuleNotFoundError: No module named 'streamlit'`:** The deploy image did not install dependencies. Ensure **root** `requirements.txt` lists all packages (not only `-r` to another path), commit, and redeploy. Check **build logs** for a successful `pip install -r requirements.txt`.
- **Build fails:** Check build logs; heavy deps (`geopandas`, etc.) can take several minutes.
- **App crashes on start:** Confirm `PORT` is not overridden; the `Procfile` passes Railway’s `PORT`.
- **DB connection errors:** Verify `DATABASE_URL`, region, and SSL; same DB must be reachable from the app service.
- **Streamlit websocket issues:** Usually fine behind Railway; if not, see [Streamlit server config](https://docs.streamlit.io/) for proxy settings.

## Files used for Railway

- `requirements.txt` (repo root) — full dependency list (what Nixpacks installs)
- `agent/requirements.txt` — `-r ../requirements.txt` for local installs
- `Procfile` → `streamlit run app.py` on `0.0.0.0:$PORT`
- `.streamlit/config.toml` → headless defaults
- `scripts/railway_pg_restore.sh` + `scripts/railway-restore.env.example` → local restore without pasting passwords into URLs
