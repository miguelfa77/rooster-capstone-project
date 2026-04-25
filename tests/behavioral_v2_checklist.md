# Rooster v2 — behavioral validation checklist (Phase 5)

Re-run the capstone 19-query set after each major agent change. Record pass/fail and notes.

## Preconditions

- `analytics` dbt models built (including `parcel_metrics`).
- `OPENAI_API_KEY` set; optional: `ROOSTER_CLASSIFIER_MODEL`, `ROOSTER_INTENT_MODEL`.
- Local Postgres with listings + neighborhood data loaded.

## Smoke checks

1. Greeting / router: one-word classifier via Responses API, no tool calls.
2. Listings: `operation` matches resolved intent (venta vs alquiler).
3. Profile: `query_neighborhood_profile` includes `output_intent`; renderer honors it with Python fallback.
4. Compare: `compare_neighborhoods` returns one aligned table.
5. Parcel: `query_parcel_metrics` against `analytics.parcel_metrics`.
6. Density chart: `query_neighborhood_density_chart` yields bar-style profile data.
7. Session memory: `session_memory_v2` updates after a data turn; `conversation_state` still populated.

## Tuning

- `ROOSTER_MAX_AGENT_STEPS` (future multi-round)
- `ROOSTER_PROMPT_CACHE` session key in `st.session_state.rooster_prompt_cache_key`
