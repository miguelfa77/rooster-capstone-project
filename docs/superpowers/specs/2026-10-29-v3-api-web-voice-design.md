# Rooster v3 — API, Web App and Voice

**Status:** Draft for approval · **Branch:** `v3` · **Date:** 2026-10-29

## 1. Intent

Rooster v2 is a working Spanish-language real estate analytics agent for Valencia, served through Streamlit. v3 keeps the agent and changes how people reach it:

1. **API** — the agent becomes a service that any client can call.
2. **Web app** — a Next.js + TypeScript frontend replaces Streamlit.
3. **Voice** — a phone number you can call in Spanish to ask the same agent questions.

**Why:** v3 is a portfolio project aimed at forward-deployed engineering roles (e.g. HappyRobot). The owner is strong in Python, data and agent work; the gaps are React/TypeScript/Node full-stack work and voice agents. v3 closes both on a product the owner already understands deeply, which makes a stronger interview story than a toy demo.

The architectural story: **one Python brain, multiple clients.** The agent stays the single source of truth; the web app and the phone line are both thin clients of the same API.

## 2. Goals, Non-Goals, Anti-Goals

**Goals**
- The web app reaches feature parity with today's Streamlit app (chat tab and Intel tab).
- Agent output streams to the browser as it is produced, as it does today.
- A caller can phone a number, ask a question in Spanish, and hear a correct spoken answer from the same agent.
- Calls are logged and visible on a page in the web app.
- Each service is independently deployable and containerized.

**Non-goals (v3)**
- User accounts, login, multi-tenancy.
- New data sources (INE, Inside Airbnb, etc.).
- Changes to the agent's reasoning: planner, semantic layer, tools, synthesiser prompts and dbt models stay as they are, except where a change is strictly needed to serve a new client (e.g. a spoken-answer mode, §5.3).

**Anti-goals — v3 has failed if:**
- The agent's logic gets duplicated or rewritten in TypeScript.
- Answer quality regresses versus v2 on the baseline eval (§5.1, Step 0).
- There is a period with no working live demo (Streamlit stays up until the web app reaches parity).
- The voice demo feels unusable because of silence: a caller should never wait in dead air.

## 3. Constraints

- **Existing stack is kept:** Python agent, OpenAI Responses API, Postgres + PostGIS, dbt, the sandbox FastAPI service.
- **Language of the product is Spanish** (UI copy, voice in and out). Code, docs and commits are in English.
- **Hosting is TBD** and must stay open. Therefore: every service ships with a Dockerfile; configuration comes only from environment variables; the frontend avoids host-specific features (no platform-only APIs), so it can run on Railway or Vercel unchanged.
- **Long responses:** an agent turn can take tens of seconds. Nothing in the request path may depend on short serverless time limits.
- **Working practices** in `.claude/CLAUDE.md` apply (Karpathy guidelines, superpowers workflow, no AI attribution in commits or PRs).

## 4. Approaches Considered

| | Approach | Trade-off |
|---|---|---|
| **A (chosen)** | Python agent behind FastAPI; Next.js + TS frontend; voice service calls the same API | Two languages and two deploys, but keeps all v2 agent work and focuses new effort on the actual gaps (TS/React, API design, voice). |
| B | Rewrite everything in TypeScript (Next.js + Node agent) | One language, most Node practice, but throws away a tested agent and roughly doubles the work. |
| C | Keep Streamlit, add FastAPI + voice only | Fastest voice demo, but skips React/TS entirely — the main gap. |

The owner chose **A**: it is the only option that closes both gaps without redoing finished work.

## 5. Design

### 5.0 Architecture

```
Browser ──► Next.js (TypeScript)          Phone ──► Telephony provider
              │  SSE / REST                              │ audio stream
              ▼                                          ▼
           FastAPI (Python, new) ◄──────────────── Voice service (Python, new)
              │  imports
              ▼
           agent/ (existing)  ──► sandbox service (existing)
              │
           Postgres + PostGIS + dbt analytics (existing)
```

### 5.1 Phase 1 — API

**Step 0: baseline eval.** The behavioral checklist refers to a 19-query set that is not in the repo. Before any refactor, a fixed set of Spanish test questions (about 20, written from the checklist and the agent's tool catalogue, covering each tool, each primitive type and a multi-turn follow-up) is committed (`tests/eval_questions.yml`) together with a script that runs each one through the agent on v2 behavior and records, per question, the tools called, the primitive types returned and the answer text (`tests/eval_baseline.json`). Every later phase is checked against this baseline. Because the LLM is non-deterministic, comparison is on tools and primitive types, with answer text reviewed by a person, not diffed. *My call:* this turns "no regression" from a judgment into something checkable, and an agent eval harness is itself a strong FDE signal.

**Step 1: extract the turn.** Today `run_agent_loop_pipeline` (`agent/agent_loop.py`) is UI-free, but the rest of a chat turn lives inside `render_chat()` in `app.py`: building conversation context, streaming synthesis (`stream_synthesize_response`), retrying failed code primitives (`_try_correct_code_primitives`, `_synthesize_with_response_review`), and updating session memory (`update_session_memory_from_turn`). This orchestration moves into a new UI-free module (e.g. `agent/turn.py`) that yields a stream of events. Streamlit's `render_chat()` is then rewritten to consume that same module, which proves the extraction preserves behavior before any new client depends on it.

**Step 2: FastAPI service** (new top-level `api/`), importing `agent/`:

- `POST /conversations` → creates a conversation, returns its id.
- `POST /conversations/{id}/messages` → runs one turn and streams **Server-Sent Events**:
  - `thought` — progress/preamble text (today's `preamble_callback`)
  - `text_delta` — streamed answer text
  - `primitive` — one finished primitive (JSON, same schema as `agent/synthesizer.py` models, including composites)
  - `follow_ups` — suggested follow-up questions
  - `error` — a user-facing failure (mirrors today's graceful fallback)
  - `done`
- `GET /conversations/{id}` → full message history with primitives, for reloading a page.
- Read endpoints for the Intel tab, backed by the same queries `app.py` uses today: market framing KPIs, neighborhood GeoJSON with metrics, map overlays (transit, VUT, listings, parcels), the opportunity ranking, and the yield-vs-liquidity data.

**Data the API sends, not pictures.** Primitives are sent as data: descriptors carry the rows they need (the current `_get_rows_for_columns` routing moves server-side, so each `primitive` event is self-contained). `code` primitives are executed in the sandbox by the API, and the event carries the result: Plotly JSON or Folium HTML. If the sandbox fails after the existing one retry, the event falls back to a `table` primitive, as today.

**Conversation state.** Per-conversation state that Streamlit keeps in `session_state` today (messages, `SessionMemoryV2`, previous execution results, previous planner response id) is stored server-side in Postgres, in a new schema (e.g. `app.conversations`, `app.messages`). *Builder's call:* exact table layout.

**Access.** No user accounts. The API is called only by the web app's server and the voice service, never directly from the browser: the browser talks to Next.js route handlers, which forward to FastAPI with a shared secret header and pass the SSE stream through. This keeps the secret off the client and avoids CORS entirely. *My call:* the alternative (browser calls FastAPI directly) would expose the API publicly with no way to protect it without accounts.

### 5.2 Phase 2 — Web App

Next.js (App Router) + TypeScript in a new top-level `web/`.

- **Chat page** — parity with today's chat tab: streamed answers, the "thinking" box, follow-up pills, message history that survives reload, and every primitive type rendered as a component:
  - `text` → markdown; `kpi` → KPI card; `table` → sortable table
  - `bar`, `scatter`, `line` → chart library
  - `choropleth`, `point_map` → map library with GeoJSON
  - `code` (Plotly JSON) → Plotly component; `code` (Folium HTML) → sandboxed iframe
  - `composite` → its children in order
- **Intel page** — parity with today's four sections: market in three figures, the Valencia map with color-by selector and layer toggles (barrios, transport, VUT, listings, parcels), opportunity ranking, and yield vs liquidity.
- **Calls page** — added in Phase 3 (§5.3).
- **UI copy in Spanish**, ported from `agent/ui_es.py`. Visual style carries over the current brand (logo and red accent).
- **Contracts are typed end to end.** The TypeScript types for API events and primitives are generated from the API's OpenAPI schema (which FastAPI derives from the existing Pydantic models), so the frontend and backend cannot drift silently.
- *Builder's call:* chart and map libraries, styling approach, component structure.

**Streamlit retirement.** Streamlit keeps running until the web app passes the parity checklist (§6), then `app.py`, `primitive_renderer.py` and Streamlit dependencies are removed in a separate change.

### 5.3 Phase 3 — Voice

A new voice service (new top-level `voice/`, Python) connects a phone number to the agent:

```
caller ⇄ telephony (phone number, audio stream)
        → streaming speech-to-text (Spanish)
        → FastAPI turn (same agent, spoken-answer mode)
        → text-to-speech (Spanish voice)
        → caller
```

- **Spoken-answer mode.** Charts and maps can't be heard, so voice turns ask the synthesiser for a short spoken answer instead of visual primitives: the key numbers, rounded and phrased for listening (e.g. "unos mil doscientos euros" rather than "1.234,56 €"), with at most a few items when listing neighborhoods. This is the one permitted agent change.
- **No dead air.** While the agent works, the caller hears a short acknowledgement (driven by the existing `thought` events) instead of silence.
- **Turn-taking.** The caller can interrupt (barge-in); the agent stops speaking and listens.
- **Hand-off line.** If the agent can't answer, it says so plainly and suggests the web app; no transfer to a human.
- **Call log.** Each call stores caller number (masked in the UI), start/end time, transcript turns, the agent's answers, and latency per turn (speech end → first audio). The web app's **Calls page** lists calls and shows each transcript with its latency numbers.
- **Tuning is part of the work, not an afterthought:** choosing the Spanish voice, transcription accuracy on Valencian neighborhood and street names (custom vocabulary from `core.neighborhoods`), and how numbers are read aloud.
- **Open, decide before Phase 3:** phone number country (a Spanish number usually needs address/ID verification; other countries are instant) and the monthly budget for the number and per-minute speech and LLM costs. Not needed for Phases 1–2.
- *Builder's call:* telephony provider, STT and TTS providers, and whether to use a voice framework (e.g. Pipecat or LiveKit Agents) or wire the stream directly. Preference: a framework that still exposes the STT → agent → TTS steps, since learning those layers is part of the goal; avoid fully hosted voice-agent platforms that hide them.

## 6. Success Criteria

**Phase 1**
- The baseline eval set and runner are committed, with v2 results recorded.
- Streamlit runs on the extracted turn module with no behavior change: the eval matches the baseline.
- An SSE turn from the API produces the same primitives as Streamlit for the same question.
- Conversation state survives an API restart.

**Phase 2 (parity checklist)**
- Every primitive type renders correctly, including composites and both `code` output types.
- Multi-turn follow-ups work (e.g. "de estos, muéstrame solo los que tienen yield > 6%" uses `filter_shown_data`).
- All four Intel sections and all five map layers work.
- Works at phone width.
- The eval matches the baseline when run through the API. Then Streamlit is removed.

**Phase 3**
- A caller can ask the eval questions marked `voice: true` by phone and get correct spoken answers.
- The caller hears audio within ~1.5 s of finishing speaking (acknowledgement or answer); latency is measured and shown on the Calls page.
- Neighborhood names in those questions are transcribed correctly.
- Interrupting the agent works.

## 7. Left to the Builder

- Exact module boundaries for the extracted turn and table layout for conversation storage.
- Frontend libraries (charts, maps, styling) and component structure.
- Voice providers and framework, within the constraint in §5.3.
- Hosting, when the owner decides (§3 keeps both Railway and Vercel possible).
- Test approach per phase, following the test-driven-development skill.
