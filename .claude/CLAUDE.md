# Rooster — Codebase Reference

**What it is:** Spanish-language AI real estate analytics chatbot for Valencia. Investors ask natural language questions; the agent queries a PostGIS database and returns dynamic visualizations and prose. Single city, single domain.

**Stack:** Python · PostgreSQL + PostGIS · dbt · OpenAI Responses API (GPT-5.5) · Streamlit · Folium · Plotly · Railway

---

## Architectural Philosophy

Rooster's agent mirrors how a frontier LLM natively operates: a reasoning model with tools, real feedback, and self-correction. No extra validation layers, no synthetic reviewers, no pre-interception of tool calls.

**The model is trusted to:**
- Understand what the user asked, in any phrasing, in Spanish
- Select the right tool and parameterise it correctly
- Read real executor errors and correct itself on the next loop step
- Decide what output format fits the question — no hardcoded word→format dictionaries

**Python's job is narrow:**
- Build SQL from tool params (model never writes SQL directly)
- Return real errors as structured results — never silently swallow them
- Render visualizations from structured descriptors the model emits

**The failure mode we actively avoid:** adding a new layer every time something breaks. When something fails, the right question is always "does the model need better information at the failing stage?" — almost never "should we add a checker on top of it?"

---

## Agent Pipeline (v2)

```
user message
  → semantic_resolver       # deterministic context enrichment — not a gatekeeper
  → planner (GPT-5.5)       # routes data|conversational, emits tool calls
  → validator               # tool name check only
  → executor                # Python builds SQL; real errors returned as tool results
  → [model self-corrects if executor fails — max 3-5 loop steps]
  → synthesiser (GPT-5.5)   # emits primitives: descriptor | text | kpi | table
  → [code primitive pre-check → sandbox → if fails, re-synthesise with error]
  → renderer                # dispatches primitives to Streamlit
  → memory_updater          # gpt-5-mini, updates structured session memory
```

---

## Key Concepts

**Semantic layer** — three YAML registries (`metrics.yml`, `concepts.yml`, `heuristics.yml`) that give the model self-knowledge about the data: what columns exist, what user-facing terms mean, how compositional concepts like "buena zona" translate to filter expressions, and what spatial references like "el centro" resolve to. The model reads these as context — not a gatekeeper.

**Visualization descriptors** — the synthesiser emits structured JSON descriptors for standard chart types instead of writing raw Python code. Each descriptor is rendered by a verified Python template with no code generation risk:
- `choropleth` — neighborhood polygon map colored by a metric (requires `geom` column)
- `bar` — horizontal/vertical bar ranking chart
- `scatter` — two-metric scatter / correlation plot
- `line` — time series / trend chart
- `point_map` — marker map for individual listings (requires `lat`/`lng`)

The `code` primitive (free-form Python via sandbox) is reserved for non-standard visualizations not covered by descriptors.

**Multi-result rendering** — each descriptor renderer uses `_get_rows_for_columns(*required)` to find the right dataset from all execution results by column presence. A choropleth needs `geom` + metric; a point_map needs `lat` + `lng`. Multi-tool responses automatically route to the correct dataset per primitive.

**Multi-turn data coherence** — `filter_shown_data` tool filters the previous turn's in-memory rows in Python with no DB query. Enables follow-up questions like "de estos, muéstrame solo los que tienen yield > 6%" to operate on already-fetched data. Previous execution results are passed to `run_agent_loop_pipeline` each turn.

**Sandbox** — a separate Railway FastAPI service that executes `code` primitives only. Uses a `ProcessPoolExecutor` with pre-warmed workers so package imports (folium, geopandas, shapely, plotly) are paid once at startup, not per request. All data injected via `df`. On failure, the synthesiser is called again with the error as a correction block (one retry), then falls back to a table.

**Session memory** — single structured object (`SessionMemoryV2`) tracking inferred user preferences, neighborhoods in focus, clarification resolutions, and conversation stage. Updated each turn by gpt-5-mini via structured outputs. Feeds into planner and synthesiser context. There is no parallel rule-based memory system.

**Executor errors as feedback** — when a tool call fails (invalid metric, bad operator, wrong format), the executor returns a structured error with the valid alternatives listed. The model reads this as a tool result and self-corrects on the next loop step. This replaces any pre-validation or reviewer layer.

---

## Data Layer (current)

Three schemas: `raw` (scrape landing) · `core` (cleaned, geocoded, spatially matched) · `analytics` (dbt views — **app only queries these**).

dbt owns the analytics schema entirely. Investment score, yield calculations, and neighborhood profiles are defined as SQL models there — auditable, version-controlled, testable.

**Primary analytics view:** `analytics.neighborhood_profile` — listing metrics, transit, tourism, investment score.

**Valid select_metrics metric names:**
```
gross_rental_yield_pct  median_venta_price  median_alquiler_price
venta_count  alquiler_count  investment_score  transit_stop_count
tourist_density_pct  median_venta_eur_per_sqm  data_confidence
```

---

## Deployment

Two Railway services: the main Streamlit app and the sandbox FastAPI service (root dir `sandbox/`, separate Dockerfile). One Railway PostGIS 17 database. Communication between app and sandbox via Railway private networking (`ROOSTER_SANDBOX_URL` — must include `http://` scheme).

**Key env vars:** `DATABASE_URL` · `OPENAI_API_KEY` · `ROOSTER_SANDBOX_URL`
