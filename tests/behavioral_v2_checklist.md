# Rooster v2 — behavioral validation checklist (post Phase F)

Re-run the capstone 19-query set after each major agent change. Record pass/fail and notes.

## Preconditions

- `analytics` dbt models built, especially `analytics.neighborhood_profile`.
- `OPENAI_API_KEY` set for live planner/synthesizer tests.
- Local Postgres with listings + neighborhood data loaded.

## Smoke checks

1. Semantic resolver: maps core user terms to canonical metrics/concepts/heuristics; unresolved essential terms clarify before planning.
2. Unified planner: greeting routes to `conversational`; data questions route to compositional tools.
3. Tool catalogue: only final tools are exposed: `select_metrics`, `compute_aggregate`, `temporal_series`, `query_listings`, `query_transit_stops`, `query_tourist_apartments`, `resolve_spatial_reference`.
4. Listings: `operation` matches user intent (venta vs alquiler); exact room counts set both `min_rooms` and `max_rooms`.
5. Reviewer: catches silent metric substitution and concept filters not applied.
6. Primitive synthesizer: returns only `text`, `kpi`, `table`, `chart`, `map`, or `composite`.
7. Chart linter: rejects invalid chart specs before rendering.
8. Session memory: `session_memory_v2` updates after a data turn; flat `conversation_state` remains populated for compatibility.

## Tuning

- `ROOSTER_MAX_AGENT_STEPS`
- `ROOSTER_MAX_AGENT_STEPS_RECOMMENDATION`
- `ROOSTER_MAX_PRIMITIVES`
- `ROOSTER_MIN_VENTA_COUNT` / `ROOSTER_MIN_ALQUILER_COUNT`
- `ROOSTER_DATA_CONFIDENCE_STRONG_MIN` / `ROOSTER_DATA_CONFIDENCE_ADEQUATE_MIN`
- `ROOSTER_PROMPT_CACHE` session key in `st.session_state.rooster_prompt_cache_key`
