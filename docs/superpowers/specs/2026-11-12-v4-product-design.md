# Rooster v4 — Product (new frontend + the backend it needs)

**Status:** Approved (owner, 2026-11-12) · **Date:** 2026-11-12 · **Branch:** `v4` (from `v3`)
**Mockups:** `docs/design/v4-mockups/` (`ticker.html` is the agreed look; `dark-light.html`, `visual-direction.html` show how we got there)

## 1. Intent

Rooster started as a proof of concept, and its UI (Streamlit, then a straight port to Next.js in v3 Phase 2) was shaped by demo limits. Rooster is now becoming a real product for **individual and institutional investors** in Valencia residential property. This project replaces the frontend with a mature product UI and adds the product features investors expect, built on **today's data**.

Order of the overall roadmap (owner's decision): **1. product (this doc) → 2. data overhaul** (fresh scheduled data, pipeline, database, analytics layer) **→ 3. agent upgrade.** The product is built against the current database through the API; the data overhaul later improves what feeds it without changing the screens. Keeping the API contract clean is what makes that possible.

The app concept is unchanged (one agent, thin clients, metrics defined once, Spanish product — see `docs/concept.md`). The UI and feature set are new.

## 2. Goals, non-goals, anti-goals

**Goals (launch scope, chosen by the owner):**
- Accounts with Google sign-in; per-user conversation history.
- Watchlist ("Seguimiento") of barrios and listings.
- Inicio: a feed of new investment opportunities.
- Barrio pages and listing pages.
- Compare 2–4 barrios.
- Deal calculator.
- Export to PDF and read-only share links.
- Ticker tape of barrio price moves and listing events.
- Chat as an always-available sidebar, expandable to full screen, toggleable; answers link into the pages.
- Light mode (default) and dark mode (toggle).

**Non-goals (for now):** workspaces/teams (individual accounts only; revisit near launch), other sign-in methods, English (Spanish only; all copy kept in one place so English is a contained job later), listing alerts/notifications, voice, scheduled scraping and any data-model overhaul (that is project 2), agent behavior changes beyond the optional page-context field (§5.9).

**Anti-goals:** a UI that still feels like a demo; agent logic, SQL or metric definitions in TypeScript; the browser holding secrets or calling FastAPI directly; features that silently show stale or wrong numbers without saying how fresh the data is.

## 3. Constraints

- Builds on v3: FastAPI `api/` (agent, conversations in `app` schema, SSE turns, published OpenAPI + generated TS types, DB-backed turn cap), `sandbox/`, PostGIS database, `docker-compose.yml`. No hosting config (owner connects Railway/Vercel later).
- Data available today: `analytics.*` dbt views (`neighborhood_profile`, `neighborhood_metrics` incl. `median_alquiler_eur_per_sqm`, `price_changes`, `listing_summary`, tourism, transport, parcels) and `core.*` (`listings` with `first_seen_at`, `last_seen_at`, `price_int_previous`; `listing_snapshots` for history; `neighborhoods`, `transit_stops`, `tourist_apartments`, `parcels`). 71 barrios. Data freshness = last manual scraper run.
- UI copy Spanish; code/docs/commits English; commits without AI attribution (repo rule).
- Valencia locale formatting (`es-ES`, grouped thousands: "1.235 €").

## 4. Approaches (decided)

| | Approach | Trade-off |
|---|---|---|
| **A (chosen)** | Fresh Next.js app on **shadcn/ui + Tailwind** (open-source accessible components we own and restyle), carrying over v3's tested plumbing (proxy, SSE parser, chat state, generated types) but none of its screens | Fastest path to a mature, consistent, accessible UI with light/dark built in; adds Tailwind + Radix |
| B | Fresh app, fully hand-built design system (like v3) | Total control, fewest deps; slow, and polish/accessibility on us — where v3 fell short |
| C | Full component kit (Mantine/MUI) | Fastest screens; generic look, heavy, hard to make feel like Rooster |

Visual direction: the owner picked the dark "pro tool" direction with Rooster red, then made **light the default** and dark the toggle (see `ticker.html`).

## 5. Design

### 5.1 Shape and navigation

```
┌──────────────────────────── ticker tape (28 px) ─────────────────────────────┐
├── left nav ──┬───────────── top bar: search · user menu ──────┬── chat ──────┤
│ Inicio       │                                                │ Rooster      │
│ Mapa         │              main area (current page)          │ sidebar      │
│ Barrios      │                                                │ (toggle,     │
│ Anuncios     │                                                │  ⤢ full      │
│ Comparar     │                                                │  screen)     │
│ Calculadora  │                                                │              │
│ Seguimiento  │                                                │              │
│ Conversaciones (recent) │                                     │              │
└──────────────┴────────────────────────────────────────────────┴──────────────┘
```

- **Routes:** `/` Inicio · `/mapa` · `/barrios` (index, sortable table of all 71) · `/barrios/[slug]` · `/anuncios` (searchable/filterable listings) · `/anuncios/[id]` · `/comparar?b=…` · `/calculadora` (optionally `?anuncio=id`) · `/seguimiento` · `/chat` and `/chat/[id]` (full-screen chat) · `/compartir/[token]` (public read-only) · `/entrar` (sign-in).
- **Global search** (top bar, also ⌘K): barrios by name and listings by address/ID; jumps to the page.
- **Phone:** left nav collapses into a menu; chat becomes a bottom sheet that can be pulled to full screen; the ticker stays (one line).
- **Data freshness** is always visible where numbers are shown ("datos a 10 nov 2026"), from the latest `last_seen_at`.

### 5.2 Look

- Light default; dark toggle; 'Sistema' available in the user menu; remembered per user.
- Rooster red (`#e5484d` family) only for primary actions and highlights; green/red for up/down; amber for warnings. Monospaced tabular figures for numbers (JetBrains Mono or similar); Inter (or similar) for text.
- shadcn/ui components themed through CSS variables for both modes. Charts (Recharts) and maps (MapLibre; CARTO light/dark basemaps per mode) restyled to match.
- PDF exports always render in light mode.

### 5.3 Chat

- One conversation, shown either in the right **sidebar** (default open; toggle remembered per user) or **full screen** (`/chat/[id]`). Switching never loses the place.
- Sidebar renders **compact** blocks: text, KPIs and small tables inline; charts as small previews; maps and wide tables as a preview card. Clicking opens the full block in a large view (dialog) or the full-screen chat.
- **Answers link into the app:** any barrio name the agent mentions that matches a known barrio becomes a link to its page; listing rows link to listing pages; when an answer is a comparison of barrios, an "Abrir comparación" action opens `/comparar` with those barrios. Linking is done by the client from data the API returns (barrio names/slugs, listing IDs in rows), never by parsing prose heuristically beyond exact barrio-name matching against the known list.
- History: conversations belong to the signed-in user; the left nav lists recent ones; full list at `/chat`. Rename and delete supported.
- Everything v3 already does stays: streaming, thinking notes, follow-up pills, busy/rate-limit notices, failure card, refetch on done.

### 5.4 Inicio — opportunities

- Main content is a **feed of opportunities** (agreed definition; tweak later):
  - venta listings only;
  - **≥ 10 % below the barrio's median €/m²** (same operation), **or** estimated gross yield **≥ 1 pp above the barrio's** gross yield;
  - **recent**: `first_seen_at` within 7 days, **or** price dropped (`price_int < price_int_previous`) within 7 days;
  - **estimated gross yield** of a listing = (barrio `median_alquiler_eur_per_sqm` × listing m² × 12) / listing price;
  - ranked by estimated yield; each card shows photo-less summary (type, rooms, m², barrio, price, €/m²), the **reason** ("12 % bajo la mediana de Benimaclet · rent. est. 7,4 %"), badge NUEVO or BAJADA, and actions: Seguir, Ver, Calcular.
- **Personal filters** (stored per user): budget range, barrios, minimum estimated yield. Empty by default (whole city).
- Above the feed: a thin strip with the city's headline numbers (median yield, spread, barrios > 6 %) and "your watchlist this week" (changes in followed items). Empty state when nothing is followed.
- Thresholds live in one server-side definition (API), not in the client.

### 5.5 Ticker tape

- Thin strip across the top of every signed-in screen, continuously scrolling; hover pauses; static under `prefers-reduced-motion`; can be hidden from the user menu.
- Items: **barrios** with median venta €/m² and change vs the previous month (`BENIMACLET 2.410 €/m² ▲ 4,1 %`), interleaved with **listing events** from the opportunities source (`NUEVO · Piso 3 hab · Benimaclet · 189.000 €`, `BAJADA · Ático · Russafa −15.000 €`). Each item links to its page.
- Monthly change is computed server-side from `core.listing_snapshots` (same source as the agent's time series). Barrios without enough data in either month are omitted.

### 5.6 Pages

- **Barrio page:** header (name, district, active listings, data date; Seguir, Comparar, Exportar PDF, Compartir); KPI row (gross yield, venta €/m², median rent, investment score with rank of 71, tourist pressure — each with 6-month change where history allows); price €/m² and rent trend (12 months); location map (barrio highlighted, listings as points); listings below median €/m² (table → listing pages); transit and tourism details; data-confidence label.
- **Listing page:** price, size, rooms, €/m² vs barrio median (with %), estimated gross yield vs barrio, price history (snapshots + previous price), first/last seen, location map with nearby transit, link to Idealista; Seguir, Calcular, Exportar PDF, Compartir.
- **Barrios index:** sortable table of all 71 (same columns as v3 ranking + yield/€/m²), with Seguir per row and multi-select → Comparar.
- **Anuncios:** searchable, filterable, paginated listing table (operation, barrio, price, m², rooms, €/m², est. yield, NUEVO/BAJADA); rows → listing pages.
- **Mapa:** full-page city map from v3 Intel (color by metric, overlays: transit, tourist flats, listings, parcels), restyled; clicking a barrio opens a side panel with its key numbers and "Ver barrio".

### 5.7 Compare and calculator

- **Comparar:** 2–4 barrios (picker with search); table of all metrics with best value highlighted per row; small multiples (bar per metric) and the trend chart overlaid; Exportar PDF, Compartir. Reachable from barrio pages, the barrios index, chat answers.
- **Calculadora:** inputs prefilled from a listing (price, m², barrio rent/m² → estimated rent) or blank; all editable:
  - purchase: price, ITP 10 % (Comunitat Valenciana, second-hand), notary + registry + gestoría (default 1.5 % of price), renovation;
  - financing: down payment % (default 20 %), interest rate (default 3.0 %), term (default 25 years) — or cash purchase;
  - income: monthly rent (default estimated), vacancy % (default 5 %);
  - yearly costs: IBI (default 0.4 % of price… editable), community fees (€/month), insurance (default 200 €/yr), maintenance (default 1 % of price).
  - outputs: gross yield, net yield, monthly cash flow, cash-on-cash return, total cash needed, mortgage payment; a simple 10-year projection chart (no appreciation by default; editable growth %).
  - calculations run in the browser from one shared, unit-tested module (pure functions); defaults live in one place. Saved calculations belong to the user (listed under Seguimiento) and can be exported/shared. Not tax advice — a one-line disclaimer.

### 5.8 Accounts, watchlist, export, share

- **Sign-in:** Google only, via **Auth.js** in the Next.js server; users and sessions stored in our Postgres (own `auth` tables). The whole app requires sign-in except `/entrar` and `/compartir/*`.
- **Invite-only (owner's decision):** only Google accounts whose email is on an allowlist can sign in. The allowlist is the server-side env var `ROOSTER_ALLOWED_EMAILS` (comma-separated, case-insensitive). Anyone else sees a clear "Rooster es solo por invitación" page. No admin UI for now. Because access is invited, the existing global per-minute turn cap stays the only usage limit (no per-user quota yet).
- **Identity to the API:** the web server forwards the signed-in user's ID to FastAPI in a header alongside the existing secret key; the API trusts it only because the key proves the caller is the web server. Every user-owned API resource is scoped by that ID; a request for another user's resource gets 404.
- **Watchlist:** follow/unfollow barrios and listings from anywhere; `/seguimiento` shows followed items with key numbers and what changed since the user last looked (last-viewed timestamp per user).
- **Export PDF:** each exportable view has a print layout; the Next.js server renders it to PDF with headless Chromium (light mode, A4, Rooster header, data date, page numbers). Exportable: barrio page, listing page, comparison, calculation, a single chat answer.
- **Share links:** creating a share snapshots the view's data at that moment and returns `/compartir/[token]` (unguessable token); anyone with the link sees a read-only page (no chat, no actions, "Hecho con Rooster" footer with sign-in link); owners can list and revoke their shares. Snapshots, not live views, so a shared analysis doesn't change under the reader.

### 5.9 Page context for the chat (last, droppable)

When the user sends a message from a barrio or listing page, the client adds an optional `context` field (`{"kind": "barrio", "id": "<slug>"}` or `{"kind": "listing", "id": "<id>"}`); the API passes it to the agent, which adds one sentence to the model's input ("El usuario está viendo el barrio Benimaclet."). No other agent change. Built as the final, separable step; a few evaluation questions cover it; removable in one place if it misbehaves.

### 5.10 API additions (on today's data)

All read endpoints serve data computed in SQL against `analytics.*`/`core.*` (new dbt views where it keeps definitions in the metric layer — e.g. monthly barrio €/m² and listing opportunity fields), published in OpenAPI so TS types stay generated:
- users & identity (header), conversations scoped per user (list, rename, delete);
- barrios: index, detail, history; listings: search, detail, history;
- opportunities feed (with user filters) and ticker items;
- watchlist CRUD (+ "changes since last viewed");
- user preferences (theme, sidebar open, ticker visible, opportunity filters);
- saved calculations CRUD;
- shares: create (snapshot), get by token (public, no user), list, revoke;
- `context` field on send-message (§5.9).
Existing `/intel/*` endpoints are reused by Mapa and Inicio's headline strip where they fit.

## 6. Delivery in phases (each shippable on its own)

1. **Foundation:** new app shell on shadcn/ui + Tailwind (nav, top bar, light/dark, chat sidebar/full-screen/sheet with v3 chat plumbing), Google sign-in, per-user conversations + history.
2. **Pages:** barrios index + barrio page, anuncios + listing page, Mapa, global search.
3. **Inicio + ticker + Seguimiento:** opportunities feed and filters, ticker tape, watchlist.
4. **Comparar + Calculadora** (with saved calculations).
5. **Export & share.**
6. **Chat links + page context.**

## 7. Testing

Same bar as v3: TDD; Python unit/API tests (+ real-Postgres store tests); web unit tests (Vitest) for logic and components; Playwright end-to-end against a fake API (extended with the new endpoints and a fake signed-in session) covering each phase's main flows at desktop and 390 px; generated-type freshness check; calculator module fully unit-tested against hand-worked examples.

## 7b. Data rules for the product (settled after the builder check)

- **Data date.** "Now" for the product is the **latest scrape** (`max(last_seen_at)`), shown as "datos a …". Every "recent" window (NUEVO, BAJADA, 7 days) and every "vs last month / 6 months" is measured back from that date, not from today, so a missed scrape doesn't empty the feed or mislabel old items.
- **Active listing** = seen in the latest scrape (`last_seen_at` within one day of the data date). Counts of "anuncios activos", the opportunities feed, the ticker events and the Anuncios list use active listings only. A listing that is no longer active keeps its page with a "Ya no está publicado (visto por última vez …)" banner; it can't be followed anew, and existing follows show it as retirado.
- **Price drops** are derived from `core.listing_snapshots` (the date the lower price first appeared), not from `price_int_previous`, which has no date.
- **Shared analytics views are not changed in this project.** Barrio medians stay as the agent sees them today (computed over all scraped listings); new product-only SQL (opportunities, ticker, histories) lives in new dbt views that reuse those medians, so the agent and the product never disagree. Restricting medians to active listings is a data-overhaul (project 2) decision.
- **History depth is unknown** until run against real data; every trend, change and ticker figure needs an explicit empty/insufficient-data state ("Sin historial suficiente"), and barrios without data in both months are left out of the ticker.
- **Identifiers.** Listings are keyed by the Idealista ID parsed from the URL (`core.listings.url` is the primary key; there is no numeric ID). Barrio slugs are derived from names (lowercase, accents stripped, hyphens), unique across the 71.
- **No district field** (no source today); the barrio header shows name, active listings and data date. District comes with the data overhaul.
- **Listing type** shown from the listing heading as scraped; no type parsing.
- **Watchlist "changes":** for followed listings — price change (from snapshots) and retired; for followed barrios — new opportunities in that barrio and the monthly €/m² change. Measured since the user's last visit to Seguimiento (one timestamp per user); no stored per-item baselines.
- **Existing v3 conversations** (Phase 1 test data, no owner) are deleted by the migration that adds `user_id NOT NULL`.

## 8. Left to the builder

Component structure; exact shadcn components; table/virtualization library choice for long lists; print-layout details; exact chart styling; caching strategy for read endpoints; the fake-session mechanism for e2e.

## 9. Calls made by the orchestrator (for owner review)

All of §7b except invite-only (owner's). Also: Branch `v4`; rebuild in place under `web/` (same service, compose unchanged; v3 screens deleted, plumbing kept); routes and nav labels above; Auth.js with own Postgres tables; user ID forwarded in a header behind the secret key; PDF via headless Chromium on the web server; share links as snapshots; calculator defaults above (editable); opportunity thresholds server-side; ticker change = month over month from listing snapshots; sign-in required for everything except share links.

**Phase 1 plan call (2026-11-12):** Auth.js uses signed-cookie (JWT) sessions instead of session rows in Postgres; users live in `app.users`, created through the API at sign-in. The web server then needs no database connection of its own.
