"""
LLM→SQL layer: natural language queries over Rooster PostgreSQL.

Uses OpenAI to generate SELECT-only SQL, validates and executes against raw/core/analytics.
"""

import json
import os
import re
import time
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
DEFAULT_PLANNER_MODEL_OPENAI = os.getenv("PLANNER_MODEL", "gpt-4o-mini")
DEFAULT_SYNTHESISER_MODEL_OPENAI = os.getenv("SYNTHESISER_MODEL", "gpt-4o")

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

### Ask Rooster — intent-specific result shapes (SQL must match so the UI can render)
- **search**: Listing rows from core.listings (always include url). JOIN core.neighborhoods n ON n.id = l.neighborhood_id; LEFT JOIN analytics.neighborhood_metrics nm ON nm.neighborhood_id = l.neighborhood_id when you need medians or below_median; LEFT JOIN analytics.neighborhood_profile np ON np.neighborhood_id = l.neighborhood_id when you need transport/tourism/investment_score. Barrio column: **n.name AS neighborhood_name**. **below_median**: venta → compare l.price_int to nm.median_venta_price; alquiler → compare to nm.median_alquiler_price (operation-aware). Never filter barrio via neighborhood_raw — use fuzzy name matching (similarity + unaccent(lower(...))) or ILIKE, never exact `=` on names.
- **compare**: Return long-format rows with columns exactly **`neighborhood_name`**, **`metric`**, **`value`** (text, text, numeric). Use **UNION ALL** blocks per metric. Example pattern (fuzzy barrio filter — user input may omit · and accents):
```sql
SELECT n.name AS neighborhood_name, 'Median €/m² (sale)' AS metric, nm.median_venta_eur_per_sqm AS value
FROM analytics.neighborhood_metrics nm
JOIN core.neighborhoods n ON n.id = nm.neighborhood_id
WHERE similarity(unaccent(lower(n.name)), unaccent(lower('russafa'))) > 0.4
   OR similarity(unaccent(lower(n.name)), unaccent(lower('natzaret'))) > 0.4
UNION ALL
SELECT n.name, 'Gross yield %', nm.gross_rental_yield_pct
FROM analytics.neighborhood_metrics nm
JOIN core.neighborhoods n ON n.id = nm.neighborhood_id
WHERE similarity(unaccent(lower(n.name)), unaccent(lower('russafa'))) > 0.4
   OR similarity(unaccent(lower(n.name)), unaccent(lower('natzaret'))) > 0.4
```
- **overview**: One (or few) row(s) of headline KPIs / metrics.
- **geo**: Either (A) **listing-level map**: rows with **lat**, **lng**, **url**, **price_int**, **area_sqm**, **neighborhood_name**, **eur_per_sqm** from core.listings joined to neighborhoods — OR (B) **neighborhood choropleth**: rows with **neighborhood_name** plus a numeric field (e.g. gross_rental_yield_pct). Prefer (A) when the user asks to show listings on a map.
- **underpriced**: Prefer **listing-level** rows with **lat**, **lng**, **price_int**, **url**, **area_sqm**, **eur_per_sqm**, **neighborhood_name**, and **neighborhood_median** (median venta for that barrio, e.g. nm.median_venta_price) OR aggregate per neighborhood as before.
- **ranking**: Rows with **neighborhood_name** and **value** (numeric — the metric being ranked, e.g. gross_rental_yield_pct, **investment_score**, or **transit_stop_count** from **analytics.neighborhood_profile**). ORDER BY value as needed. Top ~15–25 barrios.
- **trend**: Time series: a date column (**scraped_at**, **bucket_date**, or similar) and **median_price_int** (or median per period); include **neighborhood_name** when comparing multiple barrios.
- **chart**: No SQL — the app renders Plotly from **core.listings** (scatter / amenity / floor). Classifier sets **chart_type**: scatter | amenity | floor.
- **memo**: No SQL — user wants an investment memo summarising the conversation; set **sql** to null.
- **conversational**: No SQL — definitions, general knowledge, greetings, or follow-ups answerable without querying the database; set **sql** to null.
- **transit_map**: Map of **public transport stops** — SQL must query **core.transit_stops** joined to **core.neighborhoods** (fuzzy barrio filter). Return **lat**, **lng**, **name**, **stop_type**, **neighborhood_name**. Do not use listing rows for this intent.
- **tourism_map**: Map of **licensed tourist apartments (VUT)** — SQL must query **core.tourist_apartments** joined to **core.neighborhoods**. Return **lat**, **lng**, **address**, **id** (optional), **neighborhood_name**.
- **combined_map**: All three layers — set **sql** to **null** and provide **three** keys: **sql_listings** (core.listings + n.name, with url, price_int, area_sqm, eur_per_sqm, lat, lng), **sql_transit** (core.transit_stops), **sql_tourism** (core.tourist_apartments). At least one must be non-empty. Same fuzzy neighborhood filter across all three when the user names a barrio.
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

- **combined_map**: User wants listings + transit + tourism together. Set **sql** to **null**. Provide **sql_listings**, **sql_transit**, **sql_tourism** as three separate SELECT strings (each read-only). Listings query must include **url**, **price_int**, **area_sqm**, **lat**, **lng**, and eur_per_sqm as **ROUND((l.price_int::numeric / NULLIF(l.area_sqm::numeric, 0)), 0) AS eur_per_sqm**. Align neighborhood filters across the three when possible.
"""

INTENT_CONVERSATIONAL_RULES = """
INTENT: "conversational" — use this when the question:
- Asks for a definition or explanation (yield, investment score, etc.)
- Is a general knowledge question about real estate or Valencia
- Is a follow-up that can be answered from conversation history alone
- Is a greeting or meta question about what Rooster can do

When intent is "conversational", set "sql" to null.
The response will be answered directly from model knowledge and conversation context — no database query.

Do NOT use conversational when the user needs live listing counts, prices, yields, or maps from the database.

EXAMPLES:
- "What is gross rental yield?" → conversational
- "How does your investment score work?" → conversational
- "Is Valencia a good market generally?" → conversational
- "What neighborhoods did we discuss?" → conversational
- "Show me listings in Ruzafa" → search (needs data)
- "What's the yield in Natzaret?" → overview (needs data)
"""

_INTENTS = frozenset(
    {
        "search",
        "compare",
        "overview",
        "geo",
        "underpriced",
        "trend",
        "chart",
        "ranking",
        "memo",
        "conversational",
        "transit_map",
        "tourism_map",
        "combined_map",
    }
)
CHART_TYPES = frozenset({"scatter", "amenity", "floor"})

# Ask Rooster: single LLM call (intent + SQL) and summarization budgets
ASK_LLM_TIMEOUT_SEC = 30.0
SUMMARIZE_TIMEOUT_SEC = 20.0

CONFIDENCE_LEVELS = frozenset({"high", "medium", "low"})

MARKET_CONTEXT = """
VALENCIA MARKET BENCHMARKS (use these for comparative reasoning):
- Spanish national average gross rental yield: ~4.5%
- Valencia city average gross yield: ~5.2% (from your data)
- Yields above 7% in Valencia are exceptional and warrant scrutiny
- Prime neighborhoods (Ruzafa, Ciutat Vella) typically yield 3-4% due to high sale prices
- Emerging neighborhoods typically yield 6-9% with lower liquidity
- €/m² above €4,000 for venta = premium pricing in Valencia context
- €/m² below €2,000 for venta = significant discount to city average
- Monthly rent above €1,500 = upper market for Valencia
- Days on market above 60 days = weak demand signal

TRANSPORT (when analytics.neighborhood_transport / core.transit_stops exist):
- Under ~300m average distance to nearest stops from barrio centroid = excellent walkability to transit
- ~300–600m = good for most renters
- Over ~800m = weaker for renters who rely on public transport
- Metro/tram stations typically matter more than bus-only stops for long-term rental demand

TOURISM / VUT (when analytics.neighborhood_tourism exists):
- **tourist_density_pct** above ~20% = high short-term rental saturation vs residential listings — regulatory risk
- Under ~10% = more stable long-term rental market
- Valencia actively regulates tourist apartments; high-density areas may face license freezes

INVESTMENT SCORE (analytics.neighborhood_profile.investment_score):
- Rough guide: above ~8 = strong on yield + transport + low tourism pressure; ~5–8 = good with trade-offs; below ~5 = niche or weak on one dimension
- The score is explainable (yield weight + transport bonus + low-tourism bonus). Use it to support narrative, not as a black box.
"""

STAGE_INSTRUCTIONS = {
    "orienting": (
        "Give a broad market overview. Help them understand the landscape before narrowing down."
    ),
    "evaluating": (
        "Go deeper on specifics. Compare options directly. Start surfacing trade-offs."
    ),
    "deciding": (
        "Be conclusive. Synthesise what you've discussed. Surface the strongest option based on what they care about."
    ),
}

DATA_QUALITY_RULES = """
DATA QUALITY RULES:
- If venta_count or alquiler_count < 5 for a neighborhood: flag explicitly — "yield figure is based on limited listings and should be treated as indicative"
- If yield_pct IS NULL: explain why — "insufficient overlap between rent and sale listings to calculate yield"
- If only one scrape exists: note that trend data is unavailable
- Never present a figure with false confidence
"""

COMBINED_JSON_SYSTEM = (
    "You are Rooster's query planner for Valencia real estate. Reply with ONLY one JSON object with these keys: "
    "For ANY neighborhood name filter in sql: use fuzzy matching (similarity + unaccent or ILIKE), never exact n.name = '...'. "
    "In sql: never use (column DESC) or (column ASC) in SELECT — sorting is ORDER BY column DESC only. "
    "FILTER (WHERE ...) must follow AVG/COUNT/SUM, never ROUND — use ROUND(AVG(x) FILTER (WHERE ...), n) not ROUND(AVG(x)) FILTER. "
    '"intent" (search|compare|overview|geo|underpriced|trend|chart|ranking|memo|conversational|transit_map|tourism_map|combined_map), '
    '"sql" (read-only SELECT, or null for chart, memo, conversational, or combined_map), '
    'For **combined_map** only: "sql_listings", "sql_transit", "sql_tourism" (each a SELECT string, or null) — at least one required; "sql" must be null. '
    '"reasoning_focus" (short string: what you will analyse in the results — e.g. "yield spread and liquidity risk"), '
    '"caveat" (optional: one short data-quality line for the UI caption below the chart — e.g. thin listings; never repeat the analyst prose), '
    '"chart_type" (only when intent is chart: scatter | amenity | floor), '
    '"followups" (array of 2-3 short follow-up questions the user might ask next), '
    '"confidence" ("high"|"medium"|"low" — use medium/low if the question is vague or could mean several things), '
    '"interpretation" (one sentence: what you understood the user to mean). '
    "No markdown fences, no text outside the JSON."
)


NEIGHBORHOOD_NAME_MATCHING_SQL = """
NEIGHBORHOOD NAME MATCHING — CRITICAL (for SQL you plan in JSON):
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


LANGUAGE_RULES = """
LANGUAGE RULES — CRITICAL:
Never use raw column names in your response text. Translate everything to natural language:

WRONG: "transport_rating 'Excellent' y distancia ~277m"
RIGHT: "excellent transport access with stops within 300m"

WRONG: "tourist_density_pct 87.5% (tourism_pressure 'High')"
RIGHT: "87% tourist apartment density — the highest risk factor here"

WRONG: "days_on_market aparece como 0"
RIGHT: "listings are moving quickly — most appear within days of posting"

WRONG: "investment_score de 6.59"
RIGHT: "investment score of 6.6 out of 10"

Always write as an analyst speaking to a client, never as a system reading database fields aloud.
"""


OUTPUT_LANGUAGE_RULES = """
OUTPUT LANGUAGE — CRITICAL:
- Rooster's user-facing app is **Spanish**. Write the entire answer in **Spanish (español de España)** for investors.
- If the user's question is clearly and entirely in English, you may answer in English; otherwise always Spanish.
- Prefer natural Spanish: rentabilidad bruta, €/m², barrio, anuncio, vivienda turística, liquidez.
"""


FORMATTING_RULES = """
RESPONSE FORMAT RULES — follow these exactly:

1. Write ONE short paragraph maximum — 3 sentences. No more. If you need to say more, save it for follow-up turns.

2. NO markdown bold in prose. Do not use **bold** anywhere in your text response. The chart carries the visual weight.

3. Structure every response as:
   - Sentence 1: the main finding (name the top neighborhood + score or key number when relevant)
   - Sentence 2: the key trade-off or risk
   - Sentence 3: what to look at next (forward-looking, not a question)

4. If showing a ranking: mention at most 3 neighborhoods by name in the prose. The chart shows the rest.

5. Do not put data-quality caveats in this paragraph — the app may show a separate caption line below the chart.
"""


SYSTEM_PROMPT = """You are Rooster, a senior real estate analyst specialising in
Valencia, Spain. You advise property investors and developers
evaluating Valencia as an investment location.

YOUR ROLE:
You interpret query results and market context — you do not execute SQL yourself.
Every response should move the user closer to an investment decision. Lead with insight,
support with data, flag risks honestly.

YOUR VOICE:
Direct, specific, and analytical. Like a trusted advisor, not a search engine.
Never say "based on the data" or "the results show". Say what you found and what it means.

DATA YOU HAVE ACCESS TO (via the app’s queries and schema):
- core.listings: individual property listings (rent + sale) with
  price, area, rooms, floor, amenities (parking, terrace, elevator,
  AC, exterior, renovated), coordinates, neighborhood_id
- core.neighborhoods: official Valencia barrio polygons
- analytics.neighborhood_profile: THE MASTER VIEW — always prefer this when reasoning
  at barrio level. Per-neighborhood: yield, price/m², transport, tourist density,
  investment score, listing counts
- core.listing_snapshots: price history for trend queries
- core.transit_stops: public transport stops with coordinates
- core.tourist_apartments: licensed tourist apartments

KEY METRICS TO REASON ABOUT:
- Gross yield above 6% = strong by Spanish standards
- Investment score above 7 = strong across yield + transport + stability.
  Score 5–7 = good with trade-offs. Below 5 = niche only
- Tourist density above 20% = Airbnb pressure + regulatory risk
- Transport rating "Excellent" = strong access to stops (see schema for definition)
- nearest_stop_m on listings = walking distance to transit
- Price drop (price_int < price_int_previous) = motivated seller signal
- Days on market (last_seen_at - first_seen_at) = demand velocity

INVESTMENT SCORE FORMULA (be transparent if asked):
Score = (yield × 0.5) + transport bonus + low tourism bonus (see analytics.neighborhood_profile).
No black boxes — explain components when relevant.

CONVERSATION STRUCTURE:
- Short conversation (1–3 turns): orient the user, broad strokes
- Mid conversation (4–6 turns): go deeper, compare specific options
- Long conversation (7+ turns): synthesise, recommend strategies, offer memo

END EVERY RESPONSE with one forward-looking observation — not a question.
Tell them what to examine next and why.

SCHEMA JOIN PATTERN:
  core.listings l
  JOIN core.neighborhoods n ON n.id = l.neighborhood_id
  LEFT JOIN analytics.neighborhood_profile np ON np.neighborhood_id = l.neighborhood_id

NEVER use neighborhood_raw for filtering — always join spatially on neighborhood_id / n.name.

""" + NEIGHBORHOOD_NAME_MATCHING_SQL + """
CONSTRAINTS:
- Never invent figures — only what the data supports
- Never recommend a specific listing as "buy this" — compare neighborhoods and strategies

""" + OUTPUT_LANGUAGE_RULES + "\n" + LANGUAGE_RULES + "\n" + FORMATTING_RULES


# Legacy single-shot SQL path (`query()`): SQL-only, not the analyst persona.
SQL_SYSTEM_PROMPT = """You are a SQL assistant for Rooster, a Valencia real estate database.
Given a user question, generate a single PostgreSQL SELECT query. Rules:
- Only SELECT queries. No INSERT, UPDATE, DELETE, DROP, CREATE, etc.
- Use only schemas: raw, core, analytics.
- Always use schema-qualified table names (e.g. core.listings, analytics.neighborhood_metrics).
- analytics.neighborhood_metrics uses neighborhood_name (not name). core.neighborhoods uses name.
- Return ONLY the SQL, no explanation. If you wrap it in markdown, use ```sql ... ```.
- For barrio names: JOIN core.neighborhoods n ON n.id = l.neighborhood_id. Never use exact equality on n.name — use similarity(unaccent(lower(n.name)), unaccent(lower('user input'))) > 0.4 ORDER BY similarity(...) DESC, or ILIKE with unaccent(lower(...)) wildcards. Do not use neighborhood_raw for filtering.
- Limit results to 100 rows unless the user asks for more.
- When returning property listings from core.listings (or raw.listings_raw), always include the url column so results link to Idealista.
- For €/m²: `ROUND((l.price_int::numeric / NULLIF(l.area_sqm::numeric, 0)), 0)` — cast **area_sqm** to numeric so the division stays numeric and `ROUND(..., 0)` is valid.
- Do not use `l.days_on_market`; use `(l.last_seen_at::date - l.first_seen_at::date) AS days_on_market`.
- Never write `(column DESC)` or `(column ASC)` in the SELECT list — use ORDER BY ... DESC/ASC at the end of the query instead.
- FILTER (WHERE ...) must follow an aggregate (AVG, COUNT, SUM, …), never ROUND: use ROUND(AVG(x) FILTER (WHERE …), 0) not ROUND(AVG(x)) FILTER (WHERE …)."""


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


def get_market_context() -> str:
    """Benchmarks for comparative reasoning (Valencia)."""
    return MARKET_CONTEXT


def _combined_user_prompt(question: str, schema_context: str, conversation_context: str) -> str:
    conv = (conversation_context or "").strip()
    conv_block = (
        f"Previous conversation:\n{conv}\n\n" if conv else ""
    )
    return f"""You are Rooster's query planner for Valencia real estate.

{MARKET_CONTEXT}

{conv_block}Current question: {question}

Return ONLY a JSON object with:
- "intent": one of "search", "compare", "overview", "geo", "underpriced", "trend", "chart", "ranking", "memo", "conversational", "transit_map", "tourism_map", "combined_map"
- "sql": a read-only PostgreSQL SELECT, OR **null** when intent is **chart**, **memo**, **conversational**, or **combined_map**
- For **combined_map** only: include **sql_listings**, **sql_transit**, **sql_tourism** (each a full SELECT string or null). At least one must be non-empty. **sql** must be null.
- "reasoning_focus": a short internal note (e.g. "yield vs liquidity in Ruzafa vs Benimaclet") for what to focus on when interpreting the query results — not a full answer
- "caveat": optional one-line data-quality note for a small caption below the chart (e.g. scores with fewer than 5 listings are indicative). Omit or empty string if nothing notable.
- "chart_type": required when intent is **chart**: **scatter** (price vs area / scatter plot), **amenity** (amenity prevalence bars), **floor** (€/m² by floor box plot). Map phrases like "price vs area", "scatter" → scatter; "amenity", "what amenities" → amenity; "floor", "by floor" → floor.
- "followups": 2-3 short, specific follow-up questions (strings)
- "confidence": "high", "medium", or "low"
- "interpretation": one sentence describing what you understood

Schema:
{schema_context}

Rules: SELECT only, no DML, use analytics.neighborhood_metrics for aggregates,
always include url column when querying listings from core.listings or raw.listings_raw.
Use schema-qualified table names (e.g. core.listings, analytics.neighborhood_metrics).
On analytics.neighborhood_metrics always use **neighborhood_name** — never **name** (that is only on core.neighborhoods).
Limit results to 100 rows unless the user asks for more.
When selecting **eur_per_sqm** from core.listings, use
`ROUND((l.price_int::numeric / NULLIF(l.area_sqm::numeric, 0)), 0)` so `ROUND` receives **numeric** (not double precision).
Never use `l.days_on_market` — use `(l.last_seen_at::date - l.first_seen_at::date) AS days_on_market` instead.
Never write `(column DESC)` or `(column ASC)` in the SELECT list — invalid SQL; use `ORDER BY column DESC` at the end of the query.
Never write `ROUND(AVG(...)) FILTER (WHERE ...)` — use `ROUND(AVG(...) FILTER (WHERE ...), 0)` so FILTER applies to AVG.
For barrio names: JOIN core.neighborhoods n ON n.id = l.neighborhood_id; never filter with exact equality on name — use fuzzy matching (similarity + unaccent on lower(), threshold ~0.4, ORDER BY similarity DESC) or ILIKE '%'||unaccent(lower('…'))||'%' on analytics.neighborhoods / neighborhood_metrics / neighborhood_profile name columns. Never filter on neighborhood_raw alone.
If the current question is a follow-up, use the previous conversation to resolve references ("those", "same area", "sort by price instead").
Match intent to the user goal: **underpriced** = deals vs neighborhood median; **trend** = price over time from listing_snapshots; **geo** = map listings (lat/lng) or neighborhood choropleth — NOT for transport-stop-only or VUT-only maps; **transit_map** = core.transit_stops map; **tourism_map** = core.tourist_apartments map; **combined_map** = three SQLs (sql_listings, sql_transit, sql_tourism); **compare** = A vs B metrics (UNION ALL long format: neighborhood_name, metric, value); **ranking** = rank neighborhoods by a metric (neighborhood_name + value); **memo** = summarise conversation — sql null; **chart** = Plotly from listings — sql null, set chart_type; **conversational** = definitions, general Valencia/real-estate knowledge, greetings, or history-only follow-ups — sql null.

Return only valid JSON, no markdown, no explanation."""


def format_summarization_prompt(
    question: str,
    reasoning_focus: str,
    results_json: str,
    conversation_context: str,
    conversation_stage: str,
) -> str:
    stage = conversation_stage if conversation_stage in STAGE_INSTRUCTIONS else "orienting"
    stage_hint = STAGE_INSTRUCTIONS[stage]
    rf = (reasoning_focus or "").strip() or "general investment implications of the results"
    conv = (conversation_context or "").strip() or "(none)"
    return f"""You are Rooster, a senior real estate analyst for Valencia.

{MARKET_CONTEXT}

The user asked: "{question}"
Your analytical focus for this response: "{rf}"

Database results:
{results_json}

Conversation so far:
{conv}

Conversation stage: {stage} — {stage_hint}

{DATA_QUALITY_RULES}

{FORMATTING_RULES}

Write plain text only (no markdown headings, no bullet lists in the answer, no **bold**).
Write in **Spanish** unless the user's question is entirely in English.
Do not repeat data-quality caveats here if the app will show a separate caption — stay in the 3-sentence paragraph."""


def format_conversational_prompt(
    question: str,
    reasoning_focus: str,
    conversation_context: str,
) -> str:
    rf = (reasoning_focus or "").strip()
    rf_line = f"Focus (optional): {rf}\n\n" if rf else ""
    conv = (conversation_context or "").strip() or "(none)"
    return f"""You are Rooster, a senior real estate analyst for Valencia.

The user asked: "{question}"

{rf_line}Answer this directly from your expertise — no database data is available or needed for this question.

Previous conversation:
{conv}

Keep your answer to 2–3 sentences maximum.
Be specific and useful. If the question is about how Rooster works (investment score, data sources, methodology), explain it clearly and honestly.

Write in **Spanish** unless the user's question is entirely in English.
Write plain text only (no markdown headings, no bullet lists, no **bold**)."""


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


def _extract_sql(text: str) -> str | None:
    """Extract SQL from LLM response (handle ```sql blocks)."""
    text = text.strip()
    m = re.search(r"```(?:sql)?\s*([\s\S]*?)```", text, re.I)
    if m:
        return m.group(1).strip()
    if text.upper().startswith("SELECT"):
        return text
    return None


def _strip_json_fence(text: str) -> str:
    text = text.strip()
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.I)
    if m:
        return m.group(1).strip()
    return text


def _normalize_followups(raw: Any) -> list[str]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    for x in raw[:5]:
        if isinstance(x, str) and (s := x.strip()):
            out.append(s)
        if len(out) >= 3:
            break
    return out


def _reasoning_focus_from_data(data: dict[str, Any]) -> str:
    rf = data.get("reasoning_focus")
    if rf is None:
        return ""
    return str(rf).strip()


def _caveat_from_data(data: dict[str, Any]) -> str:
    c = data.get("caveat")
    if c is None:
        return ""
    return str(c).strip()


def _normalize_layer_sql(raw: Any) -> str:
    """Extract a single SELECT for combined_map sql_listings / sql_transit / sql_tourism."""
    if raw is None:
        return ""
    s = str(raw).strip()
    if not s:
        return ""
    return (_extract_sql(s) or s).strip()


def _parse_combined_json_response(raw_text: str) -> dict[str, Any]:
    """
    Returns dict with intent, sql, reasoning_focus, followups, confidence, interpretation, and optional error.
    """
    empty_meta = {
        "intent": "search",
        "sql": None,
        "reasoning_focus": "",
        "caveat": "",
        "followups": [],
        "confidence": "high",
        "interpretation": "",
    }
    if not raw_text:
        return {**empty_meta, "error": "Empty response from LLM"}
    stripped = _strip_json_fence(raw_text)
    try:
        data = json.loads(stripped)
    except json.JSONDecodeError as e:
        return {**empty_meta, "error": f"Invalid JSON from LLM: {e}"}
    intent_raw = data.get("intent") or "search"
    intent = intent_raw.strip().lower() if isinstance(intent_raw, str) else "search"
    if intent not in _INTENTS:
        intent = "search"

    conf_raw = data.get("confidence") or "high"
    conf = conf_raw.strip().lower() if isinstance(conf_raw, str) else "high"
    if conf not in CONFIDENCE_LEVELS:
        conf = "high"
    interp = data.get("interpretation") or ""
    if not isinstance(interp, str):
        interp = str(interp)
    interp = interp.strip()
    reasoning_focus = _reasoning_focus_from_data(data)
    caveat = _caveat_from_data(data)

    if intent == "memo":
        return {
            "intent": "memo",
            "sql": None,
            "reasoning_focus": reasoning_focus,
            "caveat": caveat,
            "followups": _normalize_followups(data.get("followups")),
            "confidence": conf,
            "interpretation": interp,
            "error": None,
        }

    if intent == "chart":
        ct_raw = data.get("chart_type") or "scatter"
        chart_type = ct_raw.strip().lower() if isinstance(ct_raw, str) else "scatter"
        if chart_type not in CHART_TYPES:
            chart_type = "scatter"
        sql_raw = data.get("sql")
        sql: str | None = None
        if sql_raw is not None and str(sql_raw).strip():
            sql = _extract_sql(str(sql_raw).strip()) or str(sql_raw).strip()
        return {
            "intent": "chart",
            "sql": sql,
            "reasoning_focus": reasoning_focus,
            "caveat": caveat,
            "chart_type": chart_type,
            "followups": _normalize_followups(data.get("followups")),
            "confidence": conf,
            "interpretation": interp,
            "error": None,
        }

    if intent == "conversational":
        return {
            "intent": "conversational",
            "sql": None,
            "reasoning_focus": reasoning_focus,
            "caveat": caveat,
            "followups": _normalize_followups(data.get("followups")),
            "confidence": conf,
            "interpretation": interp,
            "error": None,
        }

    if intent == "combined_map":
        sl = _normalize_layer_sql(data.get("sql_listings"))
        st = _normalize_layer_sql(data.get("sql_transit"))
        su = _normalize_layer_sql(data.get("sql_tourism"))
        if not sl and not st and not su:
            return {
                "intent": "combined_map",
                "sql": None,
                "sql_listings": None,
                "sql_transit": None,
                "sql_tourism": None,
                "reasoning_focus": reasoning_focus,
                "caveat": caveat,
                "followups": _normalize_followups(data.get("followups")),
                "confidence": "high",
                "interpretation": interp,
                "error": "combined_map requires at least one of sql_listings, sql_transit, sql_tourism",
            }
        return {
            "intent": "combined_map",
            "sql": None,
            "sql_listings": sl or None,
            "sql_transit": st or None,
            "sql_tourism": su or None,
            "reasoning_focus": reasoning_focus,
            "caveat": caveat,
            "followups": _normalize_followups(data.get("followups")),
            "confidence": conf,
            "interpretation": interp,
            "error": None,
        }

    sql_raw = data.get("sql")
    if sql_raw is None:
        return {
            "intent": intent,
            "sql": None,
            "reasoning_focus": reasoning_focus,
            "caveat": caveat,
            "followups": _normalize_followups(data.get("followups")),
            "confidence": "high",
            "interpretation": "",
            "error": "Missing sql in JSON",
        }
    sql = sql_raw.strip() if isinstance(sql_raw, str) else str(sql_raw).strip()
    if not sql:
        return {
            "intent": intent,
            "sql": None,
            "reasoning_focus": reasoning_focus,
            "caveat": caveat,
            "followups": _normalize_followups(data.get("followups")),
            "confidence": "high",
            "interpretation": "",
            "error": "Missing sql in JSON",
        }
    sql = _extract_sql(sql) or sql
    return {
        "intent": intent,
        "sql": sql,
        "reasoning_focus": reasoning_focus,
        "caveat": caveat,
        "followups": _normalize_followups(data.get("followups")),
        "confidence": conf,
        "interpretation": interp,
        "error": None,
    }


def _validate_sql(sql: str) -> tuple[bool, str]:
    """Ensure SQL is read-only."""
    sql_upper = sql.upper()
    forbidden = ("INSERT", "UPDATE", "DELETE", "DROP", "CREATE", "ALTER", "TRUNCATE", "GRANT", "REVOKE")
    for kw in forbidden:
        if kw in sql_upper:
            return False, f"Forbidden: {kw}"
    if not sql_upper.strip().startswith("SELECT"):
        return False, "Only SELECT queries allowed"
    return True, ""


def _sanitize_llm_sql(sql: str) -> str:
    """
    Fix invalid patterns the LLM sometimes emits. PostgreSQL rejects `(col DESC)` in SELECT;
    only ORDER BY may use DESC/ASC.
    """
    if not sql:
        return sql
    s = sql
    for _ in range(48):
        old = s
        s = re.sub(
            r"\(\s*([a-zA-Z_][a-zA-Z0-9_.]*)\s+DESC\s*\)",
            r"\1",
            s,
            flags=re.IGNORECASE,
        )
        s = re.sub(
            r"\(\s*([a-zA-Z_][a-zA-Z0-9_.]*)\s+ASC\s*\)",
            r"\1",
            s,
            flags=re.IGNORECASE,
        )
        if s == old:
            break
    s = _fix_round_avg_filter(s)
    return s


def _fix_round_avg_filter(sql: str) -> str:
    """
    Fix ROUND(AVG(...)) FILTER (WHERE ...) → ROUND(AVG(...) FILTER (WHERE ...), …).
    FILTER must follow the aggregate (AVG), not ROUND.
    """
    key = "ROUND(AVG("
    s = sql
    pos = 0
    for _ in range(48):
        ix = s.upper().find(key.upper(), pos)
        if ix < 0:
            break
        inner_start = ix + len(key)
        depth = 1
        j = inner_start
        while j < len(s) and depth > 0:
            if s[j] == "(":
                depth += 1
            elif s[j] == ")":
                depth -= 1
            j += 1
        if depth != 0:
            pos = ix + 1
            continue
        k = j
        while k < len(s) and s[k].isspace():
            k += 1
        if k >= len(s) or s[k] != ")":
            pos = ix + 1
            continue
        m = k + 1
        while m < len(s) and s[m].isspace():
            m += 1
        if m >= len(s) or not s[m:].upper().startswith("FILTER"):
            pos = ix + 1
            continue
        paren = s.find("(", m)
        if paren < 0:
            pos = ix + 1
            continue
        depth = 0
        p = paren
        end_filter = -1
        while p < len(s):
            if s[p] == "(":
                depth += 1
            elif s[p] == ")":
                depth -= 1
                if depth == 0:
                    end_filter = p + 1
                    break
            p += 1
        if end_filter < 0:
            pos = ix + 1
            continue
        # Always separate ")…" from FILTER (we drop ROUND's ")" so the old space before FILTER is lost)
        s = s[:ix] + s[ix:j] + " " + s[m:end_filter] + ")" + s[end_filter:]
        pos = 0
    return s


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


def _call_openai(user_question: str, model: str) -> str:
    """Call OpenAI API and return raw response text."""
    from openai import OpenAI

    client = OpenAI()
    prompt = f"{SCHEMA_DESC}\n\nUser question: {user_question}\n\nGenerate the SQL query:"
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
        # GPT-5.x and some newer models require max_completion_tokens, not max_tokens
        max_completion_tokens=1024,
    )
    if response.choices and response.choices[0].message.content:
        return response.choices[0].message.content
    return ""


def _call_openai_combined(
    user_question: str, model: str, schema_context: str, conversation_context: str = ""
) -> str:
    from openai import OpenAI

    client = OpenAI(timeout=max(ASK_LLM_TIMEOUT_SEC + 5.0, 35.0))
    prompt = _combined_user_prompt(user_question, schema_context, conversation_context)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": COMBINED_JSON_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        max_completion_tokens=1400,
    )
    if response.choices and response.choices[0].message.content:
        return response.choices[0].message.content
    return ""


def generate_intent_and_sql(
    user_question: str,
    model: str | None = None,
    progress: dict[str, Any] | None = None,
    schema_context: str | None = None,
    timeout_sec: float = ASK_LLM_TIMEOUT_SEC,
    conversation_context: str = "",
) -> dict[str, Any]:
    """
    Single LLM round-trip: JSON with intent, sql, followups, confidence, interpretation.
    Uses ``timeout_sec`` for the API call only.
    """
    t0 = time.perf_counter()
    timings_ms: dict[str, float] = {}
    schema = schema_context if schema_context is not None else SCHEMA_DESC
    conv = conversation_context or ""

    def _finish(extra: dict[str, Any]) -> dict[str, Any]:
        timings_ms["total_llm"] = round((time.perf_counter() - t0) * 1000, 1)
        out: dict[str, Any] = {
            "intent": "search",
            "sql": None,
            "reasoning_focus": "",
            "caveat": "",
            "error": None,
            "followups": [],
            "confidence": "high",
            "interpretation": "",
            "timings_ms": timings_ms,
        }
        out.update(extra)
        return out

    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_KEY")
    if not api_key:
        return _finish({"error": "OPENAI_API_KEY not set (add to agent/.env)"})
    model_name = model or os.getenv("OPENAI_MODEL", "gpt-5.4-nano")

    _set_phase(progress, "combined_llm")
    t_llm0 = time.perf_counter()

    def _llm_call() -> str:
        return _call_openai_combined(user_question, model_name, schema, conv)

    try:
        with ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(_llm_call)
            raw_text = fut.result(timeout=timeout_sec)
    except FuturesTimeoutError:
        timings_ms["llm"] = round((time.perf_counter() - t_llm0) * 1000, 1)
        return _finish({"error": f"LLM request timed out after {timeout_sec:.0f}s"})

    timings_ms["llm"] = round((time.perf_counter() - t_llm0) * 1000, 1)
    parsed = _parse_combined_json_response(raw_text)
    if parsed.get("error"):
        return _finish(
            {
                "intent": parsed["intent"],
                "sql": parsed.get("sql"),
                "reasoning_focus": parsed.get("reasoning_focus") or "",
                "caveat": parsed.get("caveat") or "",
                "error": parsed["error"],
                "followups": parsed.get("followups", []),
                "confidence": parsed.get("confidence", "high"),
                "interpretation": parsed.get("interpretation", ""),
                "chart_type": parsed.get("chart_type"),
                "sql_listings": parsed.get("sql_listings"),
                "sql_transit": parsed.get("sql_transit"),
                "sql_tourism": parsed.get("sql_tourism"),
            }
        )
    return _finish(
        {
            "intent": parsed["intent"],
            "sql": parsed["sql"],
            "reasoning_focus": parsed.get("reasoning_focus") or "",
            "caveat": parsed.get("caveat") or "",
            "error": None,
            "followups": parsed["followups"],
            "confidence": parsed["confidence"],
            "interpretation": parsed["interpretation"],
            "chart_type": parsed.get("chart_type"),
            "sql_listings": parsed.get("sql_listings"),
            "sql_transit": parsed.get("sql_transit"),
            "sql_tourism": parsed.get("sql_tourism"),
        }
    )


def execute_rooster_select(
    sql: str,
    progress: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate read-only SQL and run against PostgreSQL. Returns sql, rows, error, timings_ms."""
    t_total0 = time.perf_counter()
    timings_ms: dict[str, float] = {}

    def _finish(extra: dict[str, Any]) -> dict[str, Any]:
        timings_ms["total_execute"] = round((time.perf_counter() - t_total0) * 1000, 1)
        out: dict[str, Any] = {"sql": sql, "rows": [], "error": None, "timings_ms": timings_ms}
        out.update(extra)
        return out

    sql = _sanitize_llm_sql((sql or "").strip())

    _set_phase(progress, "parse_sql")
    t_parse0 = time.perf_counter()
    ok, err = _validate_sql(sql)
    if not ok:
        return _finish({"error": err})
    timings_ms["parse_and_validate"] = round((time.perf_counter() - t_parse0) * 1000, 1)

    _set_phase(progress, "database")
    t_db0 = time.perf_counter()
    try:
        conn = _get_conn()
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        colnames = [d[0] for d in cur.description] if cur.description else []
        cur.close()
        conn.close()
        result = [dict(zip(colnames, row)) for row in rows]
        timings_ms["database"] = round((time.perf_counter() - t_db0) * 1000, 1)
        _set_phase(progress, "done")
        return _finish({"rows": result, "error": None})
    except Exception as e:
        timings_ms["database"] = round((time.perf_counter() - t_db0) * 1000, 1)
        return _finish({"error": str(e)})


def _extract_hood_from_sql(sql: str) -> str | None:
    """Best-effort barrio token from a failed SELECT (supports FK join or legacy neighborhood_raw)."""
    for m in re.finditer(
        r"neighborhood_raw\s+ilike\s+'%([^%']+)%'",
        sql,
        re.I,
    ):
        return m.group(1)
    m = re.search(
        r"unaccent\s*\(\s*upper\s*\(\s*(?:n\.)?name\s*\)\s*\)\s*=\s*"
        r"unaccent\s*\(\s*upper\s*\(\s*'([^']+)'\s*\)\s*\)",
        sql,
        re.I,
    )
    if m:
        return m.group(1)
    m = re.search(r"(?:n\.)?name\s+ilike\s+'%([^%']+)%'", sql, re.I)
    if m:
        return m.group(1)
    return None


def generate_fallback_sql(original_sql: str) -> str | None:
    """
    Build a simple benchmark SELECT from a failed listing or neighborhood query (heuristic).
    Used when the main query returns 0 rows.
    """
    if not original_sql or not original_sql.strip():
        return None
    s = original_sql.lower()
    if "core.listings" not in s and not re.search(r"\bfrom\s+listings\b", s):
        if "neighborhood_metrics" in s or "analytics.neighborhood_metrics" in s:
            return (
                "SELECT neighborhood_name, total_count, median_venta_price, median_alquiler_price, "
                "gross_rental_yield_pct FROM analytics.neighborhood_metrics "
                "WHERE total_count > 0 ORDER BY total_count DESC LIMIT 12"
            )
        return None

    hood = _extract_hood_from_sql(original_sql)
    rooms: int | None = None
    m2 = re.search(r"rooms_int\s*=\s*(\d+)", original_sql, re.I)
    if m2:
        rooms = int(m2.group(1))

    op_venta = bool(re.search(r"\bventa\b", s))
    op_alq = bool(re.search(r"\balquiler\b", s))

    wheres: list[str] = ["l.price_int IS NOT NULL"]
    join_sql = ""
    if hood:
        hood_esc = hood.replace("'", "''")
        join_sql = "JOIN core.neighborhoods n ON n.id = l.neighborhood_id "
        wheres.append(
            f"similarity(unaccent(lower(n.name)), unaccent(lower('{hood_esc}'))) > 0.4"
        )
    if rooms is not None:
        wheres.append(f"l.rooms_int = {rooms}")
    if op_alq and not op_venta:
        wheres.append("(l.operation ILIKE '%alquiler%' OR l.operation ILIKE '%rent%')")
    elif op_venta or not op_alq:
        wheres.append("(l.operation ILIKE '%venta%' OR l.operation ILIKE '%sale%' OR l.operation ILIKE '%sell%')")

    return (
        "SELECT MIN(l.price_int) AS min_price_int, MAX(l.price_int) AS max_price_int, "
        "COUNT(*)::bigint AS how_many FROM core.listings l "
        + join_sql
        + "WHERE "
        + " AND ".join(wheres)
    )


def empty_result_fallback_context(
    original_sql: str | None,
    progress: dict[str, Any] | None = None,
) -> tuple[str | None, list[dict[str, Any]], str, dict[str, float]]:
    """
    After 0-row main query: run ``generate_fallback_sql`` and execute it.
    Returns (fallback_sql, rows, narrative_for_user, fallback_timings_ms).
    """
    fb_sql = generate_fallback_sql(original_sql or "")
    if not fb_sql:
        return None, [], "", {}
    out = execute_rooster_select(fb_sql, progress)
    fb_timing = dict(out.get("timings_ms") or {})
    if out.get("error") or not out.get("rows"):
        return (
            fb_sql,
            [],
            "No exact matches. Try widening price, rooms, or neighborhood — or ask for a nearby barrio.",
            fb_timing,
        )

    row = out["rows"][0]
    if "neighborhood_name" in row or "median_venta_price" in row:
        head = "No rows for that exact filter. Here are neighborhoods with active inventory:"
        lines = [head]
        for r in out["rows"][:5]:
            nm = r.get("neighborhood_name") or "?"
            tc = r.get("total_count")
            mv = r.get("median_venta_price")
            extra = f", median sale around €{int(mv):,}" if mv is not None else ""
            lines.append(f"· {nm}: {tc} listings{extra}.")
        return fb_sql, out["rows"], "\n\n".join(lines), fb_timing

    mins = row.get("min_price_int")
    maxs = row.get("max_price_int")
    cnt = row.get("how_many")
    bits = ["No exact matches for that search."]
    if mins is not None:
        try:
            bits.append(f"In this slice, asking prices start around €{int(mins):,}")
        except (TypeError, ValueError):
            bits.append(f"In this slice, asking prices start around €{mins}")
    if maxs is not None and maxs != mins:
        try:
            bits.append(f"up to about €{int(maxs):,}")
        except (TypeError, ValueError):
            bits.append(f"up to about €{maxs}")
    if cnt is not None:
        bits.append(f"({cnt} listings).")
    bits.append("Want me to relax the filters or show the closest matches?")
    return fb_sql, out["rows"], " ".join(bits), fb_timing


def _set_phase(progress: dict[str, Any] | None, phase: str) -> None:
    if progress is not None:
        progress["phase"] = phase


def classify_intent(
    user_question: str,
    model: str | None = None,
    progress: dict[str, Any] | None = None,
) -> str:
    """
    Fast LLM call: classify into exactly one of the six Ask intents.
    """
    _set_phase(progress, "intent_llm")
    prompt = (
        "Classify this real estate question into exactly one word: search, compare, overview, geo, underpriced, or trend. "
        f"Question: {user_question}"
    )
    system = (
        "Reply with exactly one word: search, compare, overview, geo, underpriced, or trend. "
        "No punctuation or explanation."
    )

    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_KEY")
    if not api_key:
        return "search"
    client = OpenAI()
    model_name = model or os.getenv("OPENAI_MODEL", "gpt-5.4-nano")
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        max_completion_tokens=16,
    )
    text = (response.choices[0].message.content or "").strip().lower()

    for token in text.replace(",", " ").split():
        w = token.strip(".,;:!?\"'")
        if w in _INTENTS:
            return w
    return "search"


def summarize_query_results(
    user_question: str,
    rows: list[dict[str, Any]],
    model: str | None = None,
    progress: dict[str, Any] | None = None,
    timeout_sec: float | None = SUMMARIZE_TIMEOUT_SEC,
    *,
    reasoning_focus: str = "",
    conversation_context: str = "",
    conversation_stage: str = "orienting",
) -> str:
    """
    Analyst-style summary (up to ~4 sentences) using SYSTEM_PROMPT, reasoning_focus, market benchmarks, and stage.
    ``timeout_sec`` enforced via executor. Raises ``TimeoutError`` if the API does not finish in time.
    """
    _set_phase(progress, "summarize_llm")
    # Cap payload size for the LLM
    try:
        results_str = json.dumps(rows[:25], ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        results_str = str(rows[:25])
    user_block = format_summarization_prompt(
        user_question,
        reasoning_focus,
        results_str,
        conversation_context,
        conversation_stage,
    )
    def _run() -> str:
        from openai import OpenAI

        client = OpenAI(timeout=max((timeout_sec or SUMMARIZE_TIMEOUT_SEC) + 5.0, 20.0))
        model_name = model or os.getenv("OPENAI_MODEL", "gpt-5.4-nano")
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_block},
            ],
            temperature=0,
            max_completion_tokens=500,
        )
        return (response.choices[0].message.content or "").strip()

    if timeout_sec is None:
        return _run()
    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(_run)
        try:
            return fut.result(timeout=timeout_sec)
        except FuturesTimeoutError as e:
            raise TimeoutError("Summarization timed out") from e


def summarize_conversational(
    user_question: str,
    model: str | None = None,
    progress: dict[str, Any] | None = None,
    timeout_sec: float | None = SUMMARIZE_TIMEOUT_SEC,
    *,
    reasoning_focus: str = "",
    conversation_context: str = "",
) -> str:
    """
    Short answer without database rows — definitions, general knowledge, meta questions.
    """
    _set_phase(progress, "summarize_llm")
    user_block = format_conversational_prompt(
        user_question,
        reasoning_focus,
        conversation_context,
    )
    def _run() -> str:
        from openai import OpenAI

        client = OpenAI(timeout=max((timeout_sec or SUMMARIZE_TIMEOUT_SEC) + 5.0, 20.0))
        model_name = model or os.getenv("OPENAI_MODEL", "gpt-5.4-nano")
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_block},
            ],
            temperature=0,
            max_completion_tokens=400,
        )
        return (response.choices[0].message.content or "").strip()

    if timeout_sec is None:
        return _run()
    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(_run)
        try:
            return fut.result(timeout=timeout_sec)
        except FuturesTimeoutError as e:
            raise TimeoutError("Conversational reply timed out") from e


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
        from openai import OpenAI

        client = OpenAI(timeout=max(timeout_sec + 5.0, 30.0))
        model_name = model or os.getenv("OPENAI_MODEL", "gpt-5.4-nano")
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_completion_tokens=900,
        )
        return (response.choices[0].message.content or "").strip()

    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(_run)
        try:
            return fut.result(timeout=timeout_sec)
        except FuturesTimeoutError as e:
            raise TimeoutError("Conversation memo timed out") from e


def query(
    user_question: str,
    model: str | None = None,
    progress: dict[str, Any] | None = None,
) -> dict:
    """
    Turn natural language into SQL, execute, return results.

    If ``progress`` is a dict, sets ``progress["phase"]`` to help debug timeouts:
    ``llm`` → ``parse_sql`` → ``database``.

    Returns keys:
    - sql, rows, error (as before)
    - timings_ms: {"llm", "parse_and_validate", "database", "total"} (only set when that step finished)
    """
    t_total0 = time.perf_counter()
    timings_ms: dict[str, float] = {}

    def _finish(extra: dict) -> dict:
        timings_ms["total"] = round((time.perf_counter() - t_total0) * 1000, 1)
        out = {"sql": None, "rows": [], "error": None, "timings_ms": timings_ms}
        out.update(extra)
        return out

    _set_phase(progress, "llm")
    t_llm0 = time.perf_counter()
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_KEY")
    if not api_key:
        return _finish({"error": "OPENAI_API_KEY not set (add to agent/.env)"})
    model_name = model or os.getenv("OPENAI_MODEL", "gpt-5.4-nano")
    raw_text = _call_openai(user_question, model_name)
    timings_ms["llm"] = round((time.perf_counter() - t_llm0) * 1000, 1)

    _set_phase(progress, "parse_sql")
    t_parse0 = time.perf_counter()
    if not raw_text:
        return _finish({"error": "Empty response from LLM"})
    sql = _extract_sql(raw_text)
    if not sql:
        return _finish({"sql": raw_text, "error": "Could not extract SQL from response"})

    ok, err = _validate_sql(sql)
    if not ok:
        return _finish({"sql": sql, "error": err})
    timings_ms["parse_and_validate"] = round((time.perf_counter() - t_parse0) * 1000, 1)

    _set_phase(progress, "database")
    t_db0 = time.perf_counter()
    try:
        conn = _get_conn()
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        colnames = [d[0] for d in cur.description] if cur.description else []
        cur.close()
        conn.close()
        result = [dict(zip(colnames, row)) for row in rows]
        timings_ms["database"] = round((time.perf_counter() - t_db0) * 1000, 1)
        _set_phase(progress, "done")
        return _finish({"sql": sql, "rows": result, "error": None})
    except Exception as e:
        timings_ms["database"] = round((time.perf_counter() - t_db0) * 1000, 1)
        return _finish({"sql": sql, "error": str(e)})
