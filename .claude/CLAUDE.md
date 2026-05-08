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
- Write complete Plotly or Folium Python that renders exactly what was asked for

**Python's job is narrow:**
- Build SQL from tool params (model never writes SQL directly)
- Return real errors as structured results — never silently swallow them
- Execute the model's visualization code in an isolated sandbox

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
  → synthesiser (GPT-5.5)   # reads data + user message, writes visualization code + prose
  → sandbox                 # executes Plotly/Folium Python, returns rendered output
  → renderer                # dispatches to Streamlit
  → memory_updater          # gpt-5-mini, updates structured session memory
```

---

## Key Concepts

**Semantic layer** — three YAML registries (`metrics.yml`, `concepts.yml`, `heuristics.yml`) that give the model self-knowledge about the data: what columns exist, what user-facing terms mean, how compositional concepts like "buena zona" translate to filter expressions, and what spatial references like "el centro" resolve to. The model reads these as context — they are not a gatekeeper.

**Sandbox** — a separate Railway FastAPI service that receives `{code, data}` and executes the model's Plotly or Folium Python. The model writes real code; the sandbox runs it. All data injected via `df`. No DB access, no filesystem. On failure, falls back to a table of raw data.

**Session memory** — structured object tracking inferred user preferences, neighborhoods in focus, clarification resolutions, and pending threads. Updated each turn by gpt-5-mini. Feeds into planner and synthesiser context.

**Executor errors as feedback** — when a tool call fails (invalid metric, bad operator, missing field), the executor returns a structured error with the valid alternatives listed. The model reads this as a tool result and self-corrects. This replaces any pre-validation or reviewer layer.

---

## Data Layer

Three schemas: `raw` (scrape landing) · `core` (cleaned, geocoded, spatially matched) · `analytics` (dbt views — **app only queries these**).

dbt owns the analytics schema entirely. Investment score, yield calculations, and neighborhood profiles are defined as SQL models there — auditable, version-controlled, testable.

---

## Deployment

Two Railway services: the main Streamlit app and the sandbox FastAPI service (root dir `sandbox/`, separate Dockerfile). One Railway PostGIS 17 database. Communication between app and sandbox via Railway private networking (`SANDBOX_URL`).

**Key env vars:** `DATABASE_URL` · `OPENAI_API_KEY` · `SANDBOX_URL` · `DEFAULT_SYNTHESISER_MODEL_OPENAI`