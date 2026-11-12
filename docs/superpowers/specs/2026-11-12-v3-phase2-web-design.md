# Rooster v3 Phase 2 — Web App (design addendum)

**Status:** Approved · **Branch:** `v3` · **Date:** 2026-11-12
**Parent spec:** `2026-10-29-v3-api-web-voice-design.md` (approved). This addendum settles what §5.2 and §7 left to the builder and records two scope decisions made since. Where it is silent, the parent spec holds.

## 1. Intent

Replace Streamlit with a Next.js + TypeScript web app that is a thin client of the Phase 1 API. "One Python brain, multiple clients": the browser renders data the API sends; no agent logic moves to TypeScript.

The end state of this work is the whole app runnable locally (database, sandbox, API, web) with no hosting wiring. The owner connects Railway/Vercel later.

## 2. Scope decisions (made by the owner, 2026-11-12)

- **Voice (Phase 3) is out of scope** for this round. No `voice/`, no Calls page.
- **Streamlit is removed now**, not after an eval run. The eval through the API (`run_eval compare --via-api`) cannot be run in the build environment (no OpenAI key or data). The owner runs it after; the README says how. Removal covers `app.py`, `agent/primitive_renderer.py`, Streamlit-only helpers, `.streamlit/`, the Streamlit `Procfile`, and Streamlit/Folium-only dependencies that nothing else imports.

## 3. Goals, non-goals, anti-goals

Goals: parity with the Streamlit chat tab and Intel tab (parent §5.2 and §6 Phase 2 checklist, minus the eval run); works at phone width; typed contracts generated from `api/openapi.json`; each service has a Dockerfile; a single command brings the whole stack up locally.

Non-goals: accounts, conversation list/sidebar beyond "current conversation + new conversation", dark/light theme switching, i18n beyond Spanish, any API change not strictly needed by the web app.

Anti-goals: agent logic or SQL in TypeScript; the browser holding the API key or calling FastAPI directly; the frontend re-deriving data the API already sends (e.g. re-routing rows to primitives); a frontend that only works on one host.

## 4. Approaches considered (libraries)

| | Approach | Trade-off |
|---|---|---|
| A | Plotly for all charts + Leaflet for maps | Fewest libraries, closest to v2 look; heavy Plotly bundle. |
| **B (chosen by owner)** | Recharts for bar/scatter/line + MapLibre GL for maps; Plotly only, lazy-loaded, for `code` primitives that return Plotly JSON | Lighter, idiomatic React charts and smooth WebGL maps with many parcels; one more library than A. |
| C | Recharts + Leaflet | Middle ground; still three visual libraries. |

## 5. Design

### 5.1 Shape

```
browser ── same-origin fetch ──► Next.js route handler  /api/rooster/*
                                   (adds X-Rooster-Key, streams SSE through)
                                          │
                                          ▼
                                   FastAPI (Phase 1, unchanged unless needed)
```

- `web/` — Next.js (App Router) + TypeScript. Pages: `/` (chat) and `/intel`. A top bar with the Rooster wordmark and red accent links the two.
- **Proxy:** one catch-all route handler forwards method, path, query and body to `ROOSTER_API_URL` with `X-Rooster-Key: ROOSTER_API_KEY` (both server-only env vars) and returns the upstream status and body as a stream, so SSE passes through untouched. Only the API's known path prefixes (`conversations`, `intel`) are forwarded. Runs on the Node runtime with no platform-specific APIs.
- **Types:** `openapi-typescript` generates `web/src/lib/api-types.ts` from `api/openapi.json`; a check script fails when the generated file is stale. Components import primitive/event types from there only.

### 5.2 Chat page

- On load, the current conversation id is read from `localStorage`; if present, `GET /conversations/{id}` restores history; if missing or 404, a new conversation is created on first send. "Nueva conversación" clears it.
- Sending posts the message and reads the SSE stream incrementally. Events map to UI exactly as in Streamlit: `thought` → collapsible "thinking" box; `text_delta` → streamed markdown; `primitive` → appended component; `follow_ups` → pills that send their text; `error` → the graceful-fallback card (reason + suggestion); `done` → input re-enabled. A 409 (busy) shows a short Spanish notice.
- Primitive components (one per kind, a single switch picks them): `text` (markdown), `kpi` (card), `table` (sortable; parses `rows_json`), `bar`/`scatter`/`line` (Recharts), `choropleth`/`point_map` (MapLibre, using the `data` rows the API attached), `code` with Plotly output (lazy Plotly), `code` with HTML output (sandboxed iframe, `sandbox="allow-scripts"`, no same-origin), `composite` (children in order). An unknown kind renders nothing and logs once.
- Stored history renders through the same components, so a reloaded conversation looks the same as a live one.

### 5.3 Intel page

Parity with the four Streamlit sections, using the `/intel/*` endpoints:
1. Market in three figures (KPI cards; empty-state copy when `null`).
2. Valencia map: color-by selector (the `CHOROPLETH_METRICS` options), layer toggles for barrios, transporte, VUT, anuncios, parcelas. Layers fetch lazily when first toggled on.
3. Opportunity ranking table.
4. Yield vs liquidity scatter.

### 5.4 Look

Dark shell (#0e0e0e background, #161616 cards, #2a2a2a borders, #e03030 accent, the v2 palette for transit/listing colors). Free, keyless dark raster basemap tiles (CARTO dark), attribution shown. Styling with CSS Modules plus one global stylesheet of CSS variables; no CSS framework. All UI copy in Spanish, ported from `agent/ui_es.py` into one `web/src/lib/copy.ts`.

### 5.5 Running the whole app locally

- `web/Dockerfile` (Next standalone output). Existing `api/Dockerfile` and `sandbox/Dockerfile` stay.
- A root `docker-compose.yml` brings up PostGIS, sandbox, API and web with env from a `.env` file; the README documents first-run steps (SQL migrations, loaders, `dbt run`) and how to run without Docker.
- No Railway/Vercel config files are added; existing hosting docs are kept accurate but not extended.

### 5.6 Testing

- Unit (Vitest + Testing Library): SSE parser (chunk boundaries, multi-line data), the chat stream reducer (event sequence → UI state), each primitive component against fixture payloads shaped by the generated types, the proxy (adds key, refuses unknown prefixes, streams through).
- End-to-end (Playwright): the web app against a small fake API that serves recorded event streams covering every primitive kind and the Intel endpoints; checks a full chat turn, a follow-up pill, reload restoring history, the Intel page sections and layer toggles, and phone width (390 px). Test-only code lives under `web/tests/`; no test mode in production code.
- Python suite stays green after Streamlit removal.

### 5.7 API changes this needs (the only ones)

Both are needed to serve the web client, which the parent spec permits.

1. **Published response models.** `api/openapi.json` must describe everything the web reads, or the generated types are fiction. The API declares Pydantic response models for: primitive *payloads* (each synthesizer primitive plus what `build_primitive_payload` adds: `data` rows on descriptors, `output_type` (`plotly_json` | `folium_html`) and `output` on `code`; the code→table fallback is just a `table`), `PrimitiveEvent.primitive` as that payload union, the stored message shape returned by `GET /conversations/{id}`, and each `/intel/*` response. Runtime behavior is unchanged; tests assert the exported schema contains these models and that real payload fixtures validate against them.
2. **One ordered block list per assistant message.** Today a stored message has `primitive_response` (ordered, includes text, no data) and `primitive_payloads` (data, no text), plus a `summary` that repeats KPI values. The turn additionally stores `blocks`: `build_primitive_payload` applied to every primitive in `primitive_response` order, text included. The web renders finished messages from `blocks` only, never `summary`, so nothing shows twice and order matches Streamlit's interleaving. Existing fields stay for the memory and context code that reads them.

**Live vs finished.** While a turn streams, the web shows the thinking box, streamed prose, and primitives as they arrive. On `done` it fetches `GET /conversations/{id}` and replaces the live message with the stored one, so the finished view and a reloaded view are identical by construction. Messages stored before `blocks` existed (Phase 1 test data only) render their `summary` text and `primitive_payloads`.

### 5.8 Eval after Streamlit removal

No v2 baseline was ever recorded (`tests/eval/baseline.json` does not exist). The README tells the owner to run `run_eval record` (direct, through `agent.turn`) and then `run_eval compare --via-api URL`. This checks the API against the direct turn, not against v2 — stated plainly in the README as an accepted gap.

### 5.9 Parity details settled from the repo

Thoughts are not stored; history shows no thinking box. Follow-ups on past messages render as plain text; only the latest assistant message has clickable pills. The Streamlit model/timeout settings are dropped (the API fixes both). The Intel map starts with all layers off; listings use `operation=All`. Metrics that need `neighborhood_profile` return an API error when the view is missing; the map shows the error copy instead of crashing. Chat map `geom` and Intel geometries are already GeoJSON. Root `requirements.txt` keeps anything `sandbox/` or `api/` imports.

## 6. Left to the builder

Component file layout, exact markdown and table helpers, chart styling details, MapLibre layer specifics, how the fake API is implemented.

## 7. Builder's calls in this addendum (by the orchestrator, for owner review)

CSS Modules over Tailwind (fewer moving parts); `maplibre-gl` used directly rather than a React wrapper; CARTO dark tiles; conversation id in `localStorage`; docker-compose for local whole-stack runs; Playwright against a fake API since no real keys/data exist in the build environment; published response models and stored ordered `blocks` (§5.7); refetch-on-`done` so live and reloaded views match; README eval procedure without a v2 baseline (§5.8).

**Approval:** owner approved the addendum and the §5.7–5.8 calls on 2026-11-12.
