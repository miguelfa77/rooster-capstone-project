# Rooster — Target Product After Planned Changes

This document describes what Rooster should look like after the data integration work is complete. It is the engineering and product north star, not the current state.

---

## Vision

Rooster becomes a multi-source investment intelligence platform, not just a listing search tool. The analytical foundation expands from Idealista listing data to include official socioeconomic data (INE) and real short-term rental performance data (Inside Airbnb). The agent can answer questions that no competitor can: not just "what is the yield?" but "what income level lives here, what do short-term rental operators actually earn, and what does that mean for my investment?"

---

## New Data Sources

### 1. INE — Socioeconomic Demographics

**What it provides:** Official Spanish statistics at census section (sección censal) level — average household income, population, age distribution. This is the key missing signal for investment quality: a neighborhood with high yield but low and declining income is a different risk profile than one with rising incomes.

**Key dataset:** Atlas de Distribución de Renta de los Hogares (household income by census section, updated annually). Padrón Municipal (population).

**Data warehouse flow:**

```
raw.ine_income          — raw Atlas de Renta CSV per census section
raw.ine_census          — raw Padrón population CSV per census section
        ↓ spatial join (census sections → core.neighborhoods via PostGIS)
core.neighborhood_demographics
    neighborhood_id (FK → core.neighborhoods)
    avg_household_income        — €/year, mean per census section weighted by population
    median_household_income     — €/year
    population
    population_density          — residents/km²
    pct_under_35                — share of population under 35 (rental demand signal)
    pct_over_65                 — share over 65 (neighborhood age profile)
    ine_reference_year          — vintage of the data
        ↓ dbt model
analytics.neighborhood_demographics   — or extended columns on neighborhood_profile
```

**Spatial join note:** INE publishes at sección censal granularity, which is finer than Valencia barrios. The core transformation aggregates census sections to neighborhoods using a spatial intersection weighted by area or population. This join lives in a dbt model, not in the pipeline loader.

---

### 2. Inside Airbnb — Short-Term Rental Performance

**What it provides:** Quarterly snapshots of all active Airbnb listings in Valencia — price, availability, reviews, license, neighbourhood. This gives actual short-term rental market data rather than relying on VUT license counts alone. A neighborhood's tourist_density_pct currently uses license counts vs listing counts; Inside Airbnb gives actual active supply and estimated revenue.

**Source:** insideairbnb.com — free, quarterly updated, Valencia available.

**Key files:** `listings.csv` (one row per active Airbnb listing), `calendar.csv` (daily availability — optional, heavy).

**Data warehouse flow:**

```
raw.airbnb_listings     — raw listings.csv snapshot with ingestion_date
        ↓ clean, geocode, join to neighborhoods
core.airbnb_listings
    id                          — Airbnb listing ID
    neighborhood_id (FK)        — spatially joined via lat/lng
    room_type                   — Entire home / Private room / Shared
    price_per_night             — parsed from price string (€)
    availability_365            — days available in next year (proxy for occupancy)
    estimated_occupancy_pct     — (365 - availability_365) / 365
    estimated_annual_revenue    — price_per_night × estimated_occupancy_pct × 365
    reviews_per_month
    is_licensed                 — license field non-null
    snapshot_date
        ↓ dbt model (aggregated to neighborhood)
analytics.neighborhood_airbnb
    neighborhood_id
    neighborhood_name
    airbnb_listing_count        — active entire-home listings
    airbnb_avg_nightly_price    — €/night median
    airbnb_estimated_occupancy  — median estimated occupancy %
    airbnb_median_annual_revenue — median estimated annual revenue per listing
    airbnb_licensed_pct         — share with a license
    snapshot_date
```

**Relationship to existing data:** `tourist_density_pct` currently uses VUT license counts. Inside Airbnb data is richer — it shows actual active supply including unlicensed operators, real pricing, and estimated revenue. This is the foundation for more accurate yield calculations in a future phase.

---

## Connecting New Data to the Agent

Once the warehouse layers are stable, the integration into the agent follows a fixed pattern:

**1. Semantic layer** — add new metric keys to `agent/semantic_layer/metrics.yml`:
```yaml
- key: avg_household_income
  column: analytics.neighborhood_demographics.avg_household_income
  display: Renta media del hogar
  unit: €/año
  ...
- key: airbnb_median_annual_revenue
  column: analytics.neighborhood_airbnb.airbnb_median_annual_revenue
  ...
```

**2. SQL builder** — `analytics.neighborhood_profile` either extended via a dbt join to the new models, or `build_select_metrics_sql` updated to JOIN the new analytics tables when those metrics are requested.

**3. Planner instructions** — new metric names added to the valid metrics list so the planner can emit them in `select_metrics` tool calls.

**4. Synthesiser instructions** — new Spanish display names added to `_FIELD_TEXT_REPLACEMENTS` and `_HARD_CONSTRAINTS` field name glossary.

This is the full connection path. No new tools needed — `select_metrics` handles the new metrics once the warehouse layer exists.

---

## What the Agent Can Answer After Integration

| Question | Before | After |
|----------|--------|-------|
| ¿Qué nivel de renta tiene este barrio? | ✗ No data | ✓ INE income data |
| ¿Cuánto gana un Airbnb en Russafa? | ✗ No data | ✓ Inside Airbnb revenue estimate |
| ¿Qué barrios tienen alto yield Y alta renta? | ✗ Partial | ✓ Full cross-metric analysis |
| ¿Dónde vive gente joven? | ✗ No data | ✓ INE age distribution |
| ¿Cuántos Airbnb activos tiene este barrio? | Proxy only | ✓ Actual count + licensing rate |

---

## Pipeline Architecture

Both new sources follow the same ingestion pattern as the existing pipeline:

```
pipeline/open_data/
    fetch_ine_income.py         — download Atlas de Renta CSV, load to raw.ine_income
    fetch_ine_census.py         — download Padrón CSV, load to raw.ine_census
    fetch_airbnb_listings.py    — download insideairbnb.com CSV, load to raw.airbnb_listings

sql/
    open_data_tables.sql        — DDL for new raw.* and core.* tables

dbt/models/analytics/
    neighborhood_demographics.sql   — INE aggregation to neighborhood level
    neighborhood_airbnb.sql         — Airbnb aggregation to neighborhood level
```

Loaders append with a `snapshot_date` so historical snapshots are preserved. dbt models use `WHERE snapshot_date = (SELECT MAX(snapshot_date) FROM ...)` to query the latest snapshot.

---

## Out of Scope for This Phase

- Metric recalculation (investment_score, tourist_density_pct) — deferred, requires separate design decision
- Fotocasa / other listing scrapers — separate decision
- RAG from market reports — deferred until structured data layer is solid
- Valencia city open data (schools, crime, etc.) — next phase after INE + Airbnb
