"""Database connection helpers and static schema context for Rooster.

The chat agent uses OpenAI **function calling** with fixed Python executors in
``agent/agent_pipeline.py`` — not ad-hoc SQL generation from this module.
This file still provides: ``get_schema_context()`` for the planner, PostgreSQL
connection helpers, and ``summarize_conversation_memo()`` for the investment memo.
"""

import os
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from functools import lru_cache
from pathlib import Path
from typing import Any

import psycopg2
from sqlalchemy import create_engine
from sqlalchemy.engine import URL

# Load agent/.env for API keys (GOOGLE_API_KEY, OPENAI_API_KEY, etc.)
def _load_agent_env() -> None:
    agent_dir = Path(__file__).resolve().parent
    dotenv = agent_dir / ".env"
    if not dotenv.exists():
        return
    for line in dotenv.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip("\"'")
        if k:
            os.environ[k] = v  # agent/.env overrides shell (expected for local dev)
    # OpenAI package expects OPENAI_API_KEY; support OPENAI_KEY as alias
    if os.getenv("OPENAI_KEY") and not os.getenv("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = os.environ["OPENAI_KEY"]


_load_agent_env()

# Load pipeline/.env for PG* (DB connection)
def _load_pipeline_env() -> None:
    pipeline_dir = Path(__file__).resolve().parents[1] / "pipeline"
    dotenv = pipeline_dir / ".env"
    if not dotenv.exists():
        return
    for line in dotenv.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip("\"'")
        if k and k not in os.environ:
            os.environ[k] = v


_load_pipeline_env()

# Default models when UI does not override (env-tunable)
DEFAULT_PLANNER_MODEL_OPENAI = os.getenv("PLANNER_MODEL", "gpt-5.5")
DEFAULT_SYNTHESISER_MODEL_OPENAI = os.getenv("SYNTHESISER_MODEL", "gpt-5.5")

SCHEMA_DESC = """
## Rooster database schema (PostgreSQL)

### core.listings
Current state per listing URL (PK url). Columns: url, operation, heading, price, price_int, price_int_previous, currency, period, rooms, rooms_int, area, area_sqm, floor, time_to_center, description, street_name_raw, neighborhood_raw, neighborhood_id (FK to core.neighborhoods.id when set via spatial join), scraped_at, first_seen_at, last_seen_at, geocode_quality, **nearest_stop_m** (metres to nearest transit stop; NULL if open-data not loaded).
- price_int_previous: prior price_int before the last upsert (price-change signal)
- first_seen_at / last_seen_at: timestamptz; first insert vs last scrape seen
- has_parking, has_terrace, has_elevator, is_exterior, is_renovated, has_ac, has_storage: booleans parsed from description text (regex)
- floor_int: numeric floor (0=bajo/planta baja, 99=ático, else extracted integer from floor text)
- minutes_to_center: first integer parsed from time_to_center (e.g. "12 min" → 12)
- lat, lng: WGS84 from optional geocode (may be NULL); geocode_quality is 'street' (parcel centroid), 'neighborhood' (barrio centroid), or NULL
- **Days on market**: Do **not** reference column `days_on_market` on `core.listings` in SQL — it may be missing until migrations are applied. Always compute it as `(l.last_seen_at::date - l.first_seen_at::date)` and alias if needed, e.g. `(l.last_seen_at::date - l.first_seen_at::date) AS days_on_market` (yields NULL if either timestamp is NULL). To sort by recency use `ORDER BY l.last_seen_at DESC` — never write `(l.last_seen_at DESC)` in the SELECT list.

### PostgreSQL — €/m² and `ROUND` (required)
`core.listings.area_sqm` may be **double precision**. Dividing `numeric` by it can yield **double precision**, and PostgreSQL has **no** `round(double precision, integer)` — queries then fail at runtime.
- Always compute listing €/m² with **numeric end-to-end** before `ROUND`, e.g.:
  `ROUND((l.price_int::numeric / NULLIF(l.area_sqm::numeric, 0)), 0) AS eur_per_sqm`
- Apply the same idea for any `ROUND(<expr>, n)` where `<expr>` is a ratio that might become float/double.

### core.listing_snapshots
Append-only time series: id, url, price_int, scraped_at (unique per url+scraped_at). Every load inserts one row per raw observation for trends and churn analysis.

### core.streets
Valencia city streets. Columns: id, nombre_municipio, codigo_via, street_name, tipo_via, nombre_via.

### core.neighborhoods
Neighborhoods with geometry. Columns: id, name, geom.

### Join pattern (listings ↔ barrios — always use this)
```
core.listings l
JOIN core.neighborhoods n ON n.id = l.neighborhood_id
LEFT JOIN analytics.neighborhood_metrics nm ON nm.neighborhood_id = l.neighborhood_id
```
Filter by barrio name on **n.name**, never on `l.neighborhood_raw` (scraper text only). Do not use exact `=` on names — users mistype accents and middots. Prefer `similarity(unaccent(lower(n.name)), unaccent(lower('user phrase'))) > 0.4` with `ORDER BY similarity(...) DESC`, or at minimum `unaccent(lower(n.name)) ILIKE '%' || unaccent(lower('user phrase')) || '%'`.

### core.parcels
Catastro parcels. Columns: refcat, municipio, via, numero, area, geom.

### analytics.neighborhood_metrics
Pre-aggregated by neighborhood. Columns: neighborhood_id, **neighborhood_name** (NOT `name` — that column does not exist here), geom, alquiler_count, venta_count, total_count, avg_alquiler_price, min_alquiler_price, max_alquiler_price, median_alquiler_price, avg_venta_price, min_venta_price, max_venta_price, median_venta_price, avg_alquiler_area_sqm, avg_venta_area_sqm, avg_alquiler_rooms, avg_venta_rooms, median_alquiler_eur_per_sqm, median_venta_eur_per_sqm, gross_rental_yield_pct (monthly rent × 12 / median sale × 100; NULL if missing data), avg_days_on_market (mean of `(last_seen_at::date - first_seen_at::date)` per listing; does not rely on a `days_on_market` column in SQL).
Important: `core.neighborhoods` uses column **name**; **analytics.neighborhood_metrics** uses **neighborhood_name** only. Never write `nm.name` when `nm` is neighborhood_metrics — use **nm.neighborhood_name**.

### core.transit_stops (open data)
Public transport stops (OSM / Overpass). Columns: osm_id, name, stop_type, lat, lng, geom (POINT 4326), neighborhood_id. Use for proximity queries; join to barrios via **neighborhood_id** or **ST_DWithin**.

### raw.tourist_apartments (staging)
GVA CSV rows; optional columns after geocode migration: **street_parsed**, **number_parsed**, **matched_street_code**, **geocode_quality** (`street` \| `street_centroid` \| null), **neighborhood_id**. Run **`python -m pipeline.open_data.geocode_tourist_apartments`** for street/parcel matching (Valencia **city** streets/parcels only).

### core.tourist_apartments (open data)
Licensed tourist apartments (VUT) from GVA CSV; loader filters by INE **provincia** (default **46**). Coordinates: **`ref_catastral` → `core.parcels.refcat`** when matched, else filled by **`migrate_tourist_apartments_geocode.sql`** / street-parcel pipeline. Columns: id, address, license_no, lat, lng, geom, status, neighborhood_id.

### analytics.neighborhood_transport
Per barrio: **transit_stop_count**, **avg_dist_to_stop_m** (mean distance in metres from barrio centroid to **nearest 3** citywide stops).

### analytics.neighborhood_tourism
Per barrio: **tourist_apt_count**, **tourist_density_pct** (tourist licenses vs Idealista **total_count** in that barrio).

### analytics.neighborhood_profile (prefer this for barrio questions)
Master view: listing metrics from **neighborhood_metrics** plus **transit_stop_count**, **avg_dist_to_stop_m**, **transport_rating** (Excellent/Good/Moderate/Poor), **tourist_apt_count**, **tourist_density_pct**, **tourism_pressure** (High/Moderate/Low), **investment_score** (transparent composite: yield 50% + transport bonus + low-tourism bonus). **Prefer `SELECT ... FROM analytics.neighborhood_profile`** over joining pieces manually when the view exists.

### analytics.listing_summary
Global stats by operation. Columns: operation, total, with_price, avg_price, min_price, max_price, median_price, avg_area_sqm, avg_rooms.

### analytics.price_changes
Listings with a price decrease vs previous scrape: url, neighborhood_raw, price_int, price_int_previous, price_drop_eur, price_drop_pct, last_seen_at.

### Typical query shapes (chat tools use fixed SQL; this is reference for joins/columns)
- **Listings**: `core.listings` + `JOIN core.neighborhoods n ON n.id = l.neighborhood_id`; include **url**. Optional `LEFT JOIN analytics.neighborhood_metrics nm` / `neighborhood_profile np` for medians, yield, **below_median** (venta vs `nm.median_venta_price`, alquiler vs `nm.median_alquiler_price`). Barrio label: **n.name AS neighborhood_name**. Never filter by `neighborhood_raw` alone — use fuzzy **similarity** / **ILIKE** on **n.name**.
- **Maps**: Listing dots need **lat**, **lng**, **eur_per_sqm**; **core.transit_stops** / **core.tourist_apartments** need **lat**, **lng**, **neighborhood_name** via join to **core.neighborhoods**.
- **Barrio rankings**: Prefer **analytics.neighborhood_profile** for yield, **investment_score**, transport/tourism fields.
"""

INTENT_EXAMPLES = """
INTENT CLASSIFICATION EXAMPLES:

search: "find 2-bed apartments under €200k", "show me listings in Natzaret", "what's available with parking"

compare: "compare Ruzafa and Natzaret", "how does Campanar stack up against Benimaclet", "which is better value — north or south Valencia"

overview: "give me a market summary", "how is the Valencia market", "what are the key stats"

geo: "show me on the map", "where are these listings", "map the cheapest apartments"

underpriced: "find me deals", "what's underpriced", "below median listings", "motivated sellers", "price drops"

ranking: "which neighborhoods have best yield", "rank by price per m²", "best value neighborhoods", "yield comparison across the city"

memo: "summarise what we've found", "give me a summary", "wrap up", "investment memo", "what have we concluded"

conversational: "What is gross rental yield?", "How does your investment score work?", "Is Valencia a good market generally?", "What neighborhoods did we discuss?", "Hi", "What can you do?"

transit_map: "show transport stops on a map", "paradas de metro", "mapa con las paradas de transporte", "how connected is Ruzafa", "walkability map"

tourism_map: "tourist apartments map", "apartamentos turísticos en el mapa", "VUT map", "Airbnb pressure", "short term rental density"

combined_map: "show everything on the map", "listings plus transport and tourist", "full overlay map", "all layers"
"""

INTENT_MAP_LAYER_RULES = """
MAP LAYER INTENTS (do not confuse with **geo** listing dots):

- **transit_map**: User wants **stops** (metro/bus/train), not property listings. Query **core.transit_stops t** JOIN **core.neighborhoods n ON n.id = t.neighborhood_id**. SELECT t.name, t.stop_type, t.lat, t.lng, n.name AS neighborhood_name. Filter barrio with similarity(unaccent(lower(n.name)), unaccent(lower('phrase'))) > 0.4 OR ILIKE.

- **tourism_map**: User wants **VUT / tourist licenses** on a map. Query **core.tourist_apartments ta** JOIN **core.neighborhoods n ON n.id = ta.neighborhood_id**. SELECT ta.id, ta.address, ta.lat, ta.lng, n.name AS neighborhood_name.

- **combined_map**: Listings + transit + tourism in one **LayerControl** map — the app merges multiple tool results; each layer still uses the same fuzzy barrio filter when the user names a zone.
"""

INTENT_CONVERSATIONAL_RULES = """
INTENT: "conversational" — use this when the question:
- Asks for a definition or explanation (yield, investment score, etc.)
- Is a general knowledge question about real estate or Valencia
- Is a follow-up that can be answered from conversation history alone
- Is a greeting or meta question about what Rooster can do

Answer from model knowledge and conversation context — no data tools.

Do NOT use conversational when the user needs live listing counts, prices, yields, or maps from the database.

EXAMPLES:
- "What is gross rental yield?" → conversational
- "How does your investment score work?" → conversational
- "Is Valencia a good market generally?" → conversational
- "What neighborhoods did we discuss?" → conversational
- "Show me listings in Ruzafa" → search (needs data)
- "What's the yield in Natzaret?" → overview (needs data)
"""


NEIGHBORHOOD_NAME_MATCHING_SQL = """
NEIGHBORHOOD NAME MATCHING — CRITICAL (for barrio filters and any SQL):
Valencia neighborhood names contain special characters (·, à, è, í, ï, l·l) that users will never type correctly.

NEVER generate:
  WHERE n.name = 'Sant Marceli'  -- will fail

ALWAYS use unaccent + trigram similarity (requires extension pg_trgm):
  WHERE similarity(
    unaccent(lower(n.name)),
    unaccent(lower('sant marceli'))
  ) > 0.4
  ORDER BY similarity(
    unaccent(lower(n.name)),
    unaccent(lower('sant marceli'))
  ) DESC

Or use ILIKE with wildcards as minimum:
  WHERE unaccent(lower(n.name))
    ILIKE '%' || unaccent(lower('sant marceli')) || '%'

The middle dot in Sant Marcel·lí, the accents in Benimaclet, Patraix, Marxalenes — users will not match these with exact equality.
Always use fuzzy matching for ANY filter on neighborhood name (n.name or analytics.neighborhood_name / neighborhood_profile.neighborhood_name).
"""


SQL_ORDER_SYNTAX_RULES = """
SQL SYNTAX — ORDER BY (PostgreSQL):
- NEVER put DESC or ASC inside parentheses in the SELECT list. This is invalid and causes: syntax error at or near "DESC".
  Wrong: SELECT l.url, (l.last_seen_at DESC) FROM ...
  Wrong: SELECT (l.price_int ASC) FROM ...
- Correct: SELECT l.url, l.last_seen_at FROM ... ORDER BY l.last_seen_at DESC NULLS LAST
- In SELECT, only use column expressions and casts, e.g. (l.last_seen_at::date - l.first_seen_at::date) AS days_on_market — never append DESC/ASC after a column name inside SELECT.
"""


SQL_FILTER_AGGREGATE_RULES = """
SQL SYNTAX — FILTER (PostgreSQL aggregates):
- FILTER (WHERE ...) may only follow an aggregate: AVG, COUNT, SUM, MIN, MAX, BOOL_OR, etc. ROUND is NOT an aggregate.
- WRONG: ROUND(AVG(expr)) FILTER (WHERE cond)  — FILTER attaches to ROUND → error: "FILTER specified, but round is not an aggregate function"
- RIGHT: ROUND(AVG(expr) FILTER (WHERE cond), 0)  — FILTER attaches to AVG; ROUND wraps the result.
- Same for SUM/COUNT: SUM(x) FILTER (WHERE cond), never SUM(...) then FILTER after an outer function unless that function is the aggregate.
"""

SUMMARIZE_TIMEOUT_SEC = 20.0

MEMO_SYSTEM_PROMPT = """Eres Rooster, analista inmobiliario senior para Valencia (España).
Redactas en español claro para inversores. No inventes cifras: el memo se basa solo en la conversación
que recibes. Sé concreto y profesional."""


SUMMARY_PROMPT = """Basándote en esta conversación, redacta un breve memo de inversión en **español** con esta estructura:

**Contexto de mercado**
Una frase sobre el mercado valenciano en general.

**Barrios evaluados**
Por cada barrio mencionado: nombre, rentabilidad, principal fortaleza,
principal riesgo. Una línea cada uno.

**Recomendación**
La opción más sólida según lo que el inversor priorizó en la conversación,
con una frase de justificación.

**Próximos pasos**
Dos acciones concretas que el inversor debería tomar.

Máximo ~200 palabras. Concreto y directo.

Conversación:
{full_conversation}
"""


def get_schema_context() -> str:
    """Schema text embedded in prompts; use app @st.cache_resource wrapper to avoid recomputation."""
    return (
        SCHEMA_DESC
        + "\n"
        + INTENT_EXAMPLES
        + "\n"
        + INTENT_CONVERSATIONAL_RULES
        + "\n"
        + INTENT_MAP_LAYER_RULES
        + "\n"
        + NEIGHBORHOOD_NAME_MATCHING_SQL
        + "\n"
        + SQL_ORDER_SYNTAX_RULES
        + "\n"
        + SQL_FILTER_AGGREGATE_RULES
    )


def summarize_conversation_memo(
    full_conversation: str,
    model: str | None = None,
    progress: dict[str, Any] | None = None,
    timeout_sec: float = 45.0,
) -> str:
    """Structured investment memo from chat history (no SQL)."""
    _set_phase(progress, "conversation_memo_llm")
    prompt = SUMMARY_PROMPT.format(full_conversation=full_conversation.strip() or "(empty)")

    def _run() -> str:
        from agent.responses_api import (
            extract_response_text,
            get_openai_client,
            reasoning_param_for_model,
            supports_temperature,
        )

        client = get_openai_client(max(timeout_sec, 30.0))
        model_name = model or os.getenv("OPENAI_MODEL", "gpt-5.5")
        kwargs: dict[str, Any] = {
            "model": model_name,
            "instructions": MEMO_SYSTEM_PROMPT,
            "input": prompt,
            "max_output_tokens": 900,
        }
        if supports_temperature(model_name):
            kwargs["temperature"] = 0.2
        rpar = reasoning_param_for_model(model_name, "low")
        if rpar is not None:
            kwargs["reasoning"] = rpar
        response = client.responses.create(**kwargs)
        return extract_response_text(response).strip()

    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(_run)
        try:
            return fut.result(timeout=timeout_sec)
        except FuturesTimeoutError as e:
            raise TimeoutError("Conversation memo timed out") from e


def _set_phase(progress: dict | None, phase: str) -> None:
    if progress is not None:
        progress["phase"] = phase


def _normalize_database_url_for_sqlalchemy(dsn: str) -> str:
    """Railway/Heroku use ``postgres://``; SQLAlchemy expects ``postgresql://`` as the dialect name."""
    dsn = dsn.strip()
    if dsn.startswith("postgres://"):
        return "postgresql://" + dsn[len("postgres://") :]
    return dsn


def _get_conn() -> Any:
    dsn = os.getenv("DATABASE_URL")
    if dsn:
        return psycopg2.connect(dsn)
    return psycopg2.connect(
        host=os.getenv("PGHOST", "localhost"),
        port=os.getenv("PGPORT", "5432"),
        user=os.getenv("PGUSER", "postgres"),
        password=os.getenv("PGPASSWORD", ""),
        dbname=os.getenv("PGDATABASE", "rooster"),
    )


def get_pg_conn() -> Any:
    """New PostgreSQL connection using DATABASE_URL or PG* (same as query execution)."""
    return _get_conn()


@lru_cache(maxsize=1)
def get_pg_engine():
    """SQLAlchemy engine for pandas ``read_sql*`` (avoids DBAPI2 UserWarning on raw psycopg2)."""
    dsn = os.getenv("DATABASE_URL")
    if dsn:
        return create_engine(
            _normalize_database_url_for_sqlalchemy(dsn), pool_pre_ping=True
        )
    return create_engine(
        URL.create(
            drivername="postgresql+psycopg2",
            username=os.getenv("PGUSER", "postgres"),
            password=os.getenv("PGPASSWORD", ""),
            host=os.getenv("PGHOST", "localhost"),
            port=int(os.getenv("PGPORT", "5432")),
            database=os.getenv("PGDATABASE", "rooster"),
        ),
        pool_pre_ping=True,
    )
