"""
Ask Rooster agent: OpenAI function calling → validate → execute → (optional) synthesiser fallback.
Pure-Python validation, execution, and renderer overrides.
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
import time
import unicodedata
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from difflib import SequenceMatcher
from typing import Any, Callable, Iterator

import pandas as pd
from sqlalchemy import text

from agent.openai_tools import (
    DEFAULT_RENDERER_FOR_TOOL,
    get_rooster_openai_tools,
)
from agent.llm_sql import (
    DEFAULT_SYNTHESISER_MODEL_OPENAI,
    SUMMARIZE_TIMEOUT_SEC,
    get_pg_engine,
)

_LOG = logging.getLogger("rooster.agent")

PLANNER_PREAMBLE_NOTE = """
=== TOOL PREAMBLES ===
Before calling any tool, output one **short Spanish sentence** in plain text (no tool call) saying what
you are checking, e.g. "Compruebo rendimientos en Russafa…". The UI may show it while tools run.
"""

CONTEXT_RESOLUTION = """
=== RESOLVING REFERENCES IN FOLLOW-UP QUESTIONS ===

When the user uses words like:
"estos" / "these" / "ellos" / "them" / "los mismos" / "those"
"este" / "this one" / "el que mencionaste"
"enséñamelos" / "show me them" / "muéstramelos"

YOU MUST resolve what they refer to before planning.

Resolve the referenced objects from the "Last assistant response" block (prose + displayed data).
Use exact barrio names, URLs, or prior filters from that block when planning the next tool call.

EXAMPLE:
Previous response showed: ranking of Sant Marcel·lí, Els Orriols, Sant Isidre, Natzaret, Tres Forques
User says: "enséñame un mapa con estos barrios"

CORRECT: call `query_neighborhood_profile` with params.neighborhoods listing those five names;
set params.output_intent = "map_neighborhoods" so the UI renders barrio polygons.

WRONG: switching to `query_listings` with empty params or `map_listings` when the user clearly refers to barrios from the last turn.

NEVER ignore a reference word. Always resolve it explicitly.
If you cannot determine what it refers to, reply briefly without tools asking: "¿A qué barrios te refieres exactamente?" (or equivalent).
=== END CONTEXT RESOLUTION ===
"""

OPENAI_TOOLS_FIRST_SYSTEM_PREFIX = """You are Rooster's planning assistant for Valencia real estate. You may call the provided functions to fetch real data, or reply briefly without calling tools when no database is needed.

LANGUAGE: Rooster's UI is Spanish. If you reply without tools (greetings, meta questions), write in **Spanish (español)** unless the user's message is clearly entirely in English.

""" + PLANNER_PREAMBLE_NOTE + "\n" + CONTEXT_RESOLUTION + """

When to call tools:
- Questions about listings, yields, neighborhoods, transport, tourist apartments, price trends, or charts → call the appropriate function(s).
- Every tool call MUST include `output_intent`; this is the model's display commitment. Prefer explicit values (e.g. `map_listings`, `map_neighborhoods`, `bar_chart`, `table`) over `auto`.

When NOT to call tools:
- Greetings, thanks, small talk, or meta questions about what Rooster can do → reply in a short conversational way (no tools, max ~3 sentences).

If you call tools, do not write prose in the first turn; use tool calls only.

**Room filters (query_listings):** Spanish phrases like "2 habitaciones" / "3 dormitorios" usually mean **exactly** that many — set **both** min_rooms and max_rooms to N. Use only min_rooms when the user clearly wants a **floor** ("al menos 2 habitaciones", "mínimo 3 dormitorios", "at least 2 bedrooms").
"""

# --- Fast path (before full FC + schema): tiny router + canned replies ---
CLASSIFIER_TIMEOUT_SEC = 3.0

CONVERSATIONAL_CLASSIFIER_PROMPT = """
You are a router for a real estate assistant called Rooster.

Classify the user message as EXACTLY one word:
- "conversational" — greetings, thanks, small talk, meta questions
  about what Rooster can do, acknowledgements, anything that needs
  zero database data
- "data" — anything about properties, neighborhoods, prices, yield,
  transport, investment, listings, maps, charts

Examples:
"hola" → conversational
"hola que tal" → conversational
"gracias" → conversational
"qué puedes hacer" → conversational
"quien eres" → conversational
"ok perfecto" → conversational
"quiero invertir en natzaret" → data
"mejor yield en valencia" → data
"compara ruzafa y benimaclet" → data
"muéstrame el mapa de transporte" → data
"hola, busco piso en ruzafa" → data

Return only one word. No explanation.
"""

CONVERSATIONAL_RESPONSES: tuple[str, ...] = (
    "Hola. Soy Rooster, tu analista del mercado inmobiliario de Valencia. ¿Buscas oportunidades de inversión, quieres comparar barrios, o te interesa ver qué hay disponible en una zona concreta?",
    "Hey. Pregúntame lo que necesites sobre el mercado de Valencia — yields, barrios, anuncios, transporte. ¿Por dónde empezamos?",
    "Hola. Tengo datos de miles de anuncios en Valencia, yields por barrio, paradas de transporte y viviendas turísticas. ¿Qué quieres analizar?",
)

META_RESPONSE = (
    "Soy Rooster — un analista inmobiliario para el mercado de Valencia. Puedo ayudarte a encontrar barrios con mejor yield, comparar zonas, ver anuncios disponibles, analizar presión turística o conectividad de transporte. ¿Qué te interesa?"
)

# Shorter cap when FC path returns conversational without a precomputed reply.
CONVERSATIONAL_SYNTH_MAX_TOKENS = 600

_META_PHRASES: tuple[str, ...] = (
    "qué puedes",
    "que puedes",
    "cómo funciona",
    "como funciona",
    "quien eres",
    "quién eres",
    "what can you",
    "how do you work",
    "what do you do",
)


def is_conversational_message(
    question: str,
    timeout_sec: float = CLASSIFIER_TIMEOUT_SEC,
) -> bool:
    """
    Fast router: one cheap completion, no tools/schema/DB.
    Returns True → caller may use canned conversational reply.
    On any failure, returns False (full pipeline).
    """
    if not (os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_KEY")):
        return False
    q = (question or "").strip()
    if not q:
        return False

    def _call() -> str:
        from agent.responses_api import responses_classify_conversational

        return responses_classify_conversational(
            q,
            system_prompt=CONVERSATIONAL_CLASSIFIER_PROMPT,
            timeout_sec=timeout_sec,
        )

    try:
        with ThreadPoolExecutor(max_workers=1) as ex:
            raw = ex.submit(_call).result(timeout=timeout_sec)
    except Exception:
        return False

    first = raw.split()[0] if raw else ""
    return first == "conversational" or raw.startswith("conversational")


def pick_fast_path_conversational_reply(question: str) -> str:
    """Canned reply for fast path; meta questions get META_RESPONSE."""
    q_lower = (question or "").lower().strip()
    if any(p in q_lower for p in _META_PHRASES):
        return META_RESPONSE
    return random.choice(CONVERSATIONAL_RESPONSES)


GROUNDED_ACK_REPLIES: tuple[str, ...] = (
    "De nada.",
    "Encantado de ayudar.",
    "Para eso estamos.",
)

_CHIT_CHAT_TOKENS = frozenset(
    {
        "hola",
        "gracias",
        "thanks",
        "thank",
        "you",
        "ok",
        "vale",
        "perfecto",
        "hi",
        "hey",
        "genial",
        "bueno",
    }
)

# Only scan recent turns so a follow-up is not "grounded" to ancient tool UI forever.
_GROUNDED_RECENT_MESSAGE_WINDOW = 10


def _is_thanks_only(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return False
    return bool(
        re.match(
            r"^(gracias|thanks|thank you|ty|muchas gracias|thank)\s*!?\s*$",
            t,
        )
    )


def _is_pure_chit_chat_message(text: str) -> bool:
    """Short acknowledgements / greetings — not follow-ups about data or prior results."""
    t = (text or "").strip().lower()
    if not t:
        return True
    t = re.sub(r"[!?.]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    words = [w for w in t.split() if w]
    if not words:
        return True
    if len(words) <= 4 and all(w in _CHIT_CHAT_TOKENS or len(w) <= 2 for w in words):
        return True
    return False


def last_assistant_message_had_tool_ui(messages: list[dict[str, Any]]) -> bool:
    """
    True if a recent assistant turn (before the current user message) showed tool-backed UI.
    """
    if len(messages) < 2:
        return False
    prior = messages[:-1]
    recent = (
        prior[-_GROUNDED_RECENT_MESSAGE_WINDOW:]
        if len(prior) > _GROUNDED_RECENT_MESSAGE_WINDOW
        else prior
    )
    for m in reversed(recent):
        if m.get("role") != "assistant":
            continue
        if m.get("agent_turn") and (m.get("render_stack") or []):
            return True
    return False


def use_conversational_fast_path(messages: list[dict[str, Any]], user_input: str) -> bool:
    """
    Whether to use the cheap canned conversational path (no FC / DB).
    After a recent tool-backed reply, substantive follow-ups use the full agent.
    Only a bare thanks uses the fast path (short ack, not a random intro).
    """
    if last_assistant_message_had_tool_ui(messages):
        if _is_thanks_only(user_input):
            return True
        return False
    if _is_pure_chit_chat_message(user_input):
        return True
    return is_conversational_message(user_input)


def pick_conversational_reply(
    messages: list[dict[str, Any]],
    user_input: str,
) -> str:
    """Reply for the conversational fast path (thanks after data vs cold open)."""
    if last_assistant_message_had_tool_ui(messages) and _is_thanks_only(user_input):
        return random.choice(GROUNDED_ACK_REPLIES)
    return pick_fast_path_conversational_reply(user_input)


def stream_canned_text_word_by_word(
    text: str,
    delay_sec: float = 0.03,
) -> Iterator[str]:
    """Yield words for st.write_stream so canned text feels streamed."""
    words = (text or "").split()
    if not words:
        return
    for i, w in enumerate(words):
        yield w + (" " if i < len(words) - 1 else "")
        if delay_sec > 0 and i < len(words) - 1:
            time.sleep(delay_sec)


SYNTHESISER_SYSTEM_PROMPT = """You are Rooster, a senior real estate analyst for Valencia, Spain.

LANGUAGE (mandatory):
- The product is used in **Spanish**. Write **all** text the user will read in **Spanish (español de España)** — clear, professional, natural for a Spanish investor.
- Match the user's language only if they write **fully in English**; for Spanish or mixed Spanish/English questions, answer in Spanish.
- Use Spanish terms: rentabilidad bruta, €/m², barrio, anuncio, vivienda turística, transporte, etc. Do not answer in English by default.

YOUR VOICE:
- Direct and specific. Name neighborhoods, quote actual numbers.
- Never use raw database column names in prose — translate to natural language.
- Never say "based on the data" or "the results show".
- No bullet points. No markdown bold in prose.
- Maximum 3 sentences.
- End with one forward-looking observation (not a question).

The user message includes which visual(s) appear after your text. Your prose MUST match those visuals (e.g. do not say "map" if the visual is a table or chart only).

If there was no execution results (conversational tool_calls empty), answer from expertise and conversation context only.

If tools failed or returned 0 rows, say so honestly and suggest what to try next.

FOLLOW-UP PILLS (required when tools ran and returned something to show):
After your 3 sentences, output a NEW LINE, then exactly ONE HTML comment:
<!-- FOLLOW_UPS: ["acción corta 1", "acción corta 2", "acción corta 3"] -->
The payload MUST be a JSON **array of strings** only (not an object, not "suggestions", not nested JSON).
2-3 strings, max 8 words each, **Spanish**, specific to what was just shown.
Examples: after a map → "Ver tabla de estos anuncios"; after a ranking → "Comparar los dos mejores barrios".
If there is nothing to build on (error-only, or pure chit-chat), use: <!-- FOLLOW_UPS: [] -->
Nothing may appear after the closing `-->` — the comment must be the last characters of your reply."""


def _sql_escape(s: str) -> str:
    return (s or "").replace("'", "''")


def get_live_schema_context() -> str:
    """Build live DB snapshot text for the planner (call from app with @st.cache_data)."""
    engine = get_pg_engine()
    lines: list[str] = [
        "=== LIVE DATABASE STATE ===",
        "TOTAL DATA:",
    ]
    try:
        counts_df = pd.read_sql(
            text(
                """
                SELECT
                    (SELECT COUNT(*) FROM core.listings WHERE price_int > 0) AS total_listings,
                    (SELECT COUNT(*) FROM core.transit_stops) AS transit_stops,
                    (SELECT COUNT(*) FROM core.tourist_apartments
                     WHERE status IS NULL OR COALESCE(lower(status), '') = 'active') AS tourist_apts,
                    (SELECT COUNT(DISTINCT neighborhood_id) FROM core.listings WHERE neighborhood_id IS NOT NULL)
                        AS neighborhoods_with_listings
                """
            ),
            engine,
        ).iloc[0]
        lines.append(f"  Listings: {int(counts_df['total_listings']):,}")
        lines.append(f"  Transit stops: {int(counts_df['transit_stops']):,}")
        lines.append(f"  Tourist apartments (active or unset status): {int(counts_df['tourist_apts']):,}")
        lines.append(f"  Neighborhoods with listings: {int(counts_df['neighborhoods_with_listings'])}")
    except Exception as e:
        lines.append(f"  (counts unavailable: {e})")

    lines.extend(["", "NEIGHBORHOODS (with listings — exact names as stored):"])
    try:
        nb = pd.read_sql(
            text(
                """
                SELECT
                    n.name,
                    COUNT(l.url) FILTER (WHERE l.operation = 'venta') AS venta_count,
                    COUNT(l.url) FILTER (WHERE l.operation = 'alquiler') AS alquiler_count,
                    BOOL_OR(np.gross_rental_yield_pct IS NOT NULL) AS has_yield
                FROM core.neighborhoods n
                LEFT JOIN core.listings l
                    ON l.neighborhood_id = n.id AND l.price_int > 0
                LEFT JOIN analytics.neighborhood_profile np ON np.neighborhood_id = n.id
                GROUP BY n.id, n.name
                HAVING COUNT(l.url) > 0
                ORDER BY COUNT(l.url) DESC
                LIMIT 80
                """
            ),
            engine,
        )
        for _, row in nb.iterrows():
            flags: list[str] = []
            if int(row["venta_count"] or 0) > 0:
                flags.append(f"{int(row['venta_count'])} venta")
            if int(row["alquiler_count"] or 0) > 0:
                flags.append(f"{int(row['alquiler_count'])} alquiler")
            if row.get("has_yield"):
                flags.append("has yield")
            lines.append(f"  - {row['name']} ({', '.join(flags)})")
    except Exception as e:
        lines.append(f"  (neighborhood list unavailable: {e})")

    lines.extend(["", "PRICE RANGES (listings, price_int > 0):"])
    try:
        prices_df = pd.read_sql(
            text(
                """
                SELECT operation,
                       MIN(price_int) AS min_price,
                       MAX(price_int) AS max_price,
                       PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY price_int) AS median_price
                FROM core.listings
                WHERE price_int > 0
                GROUP BY operation
                """
            ),
            engine,
        )
        for _, row in prices_df.iterrows():
            op = row["operation"]
            lines.append(
                f"  {op}: €{int(row['min_price']):,} – €{int(row['max_price']):,} "
                f"(median €{int(row['median_price']):,})"
            )
    except Exception as e:
        lines.append(f"  (prices unavailable: {e})")

    lines.extend(
        [
            "",
            "IMPORTANT: When the user names a barrio, match fuzzy to the list above and use the exact DB name in tool params.",
            "=== END DATABASE STATE ===",
        ]
    )
    return "\n".join(lines)


def extract_neighborhood_names_from_schema(schema_context: str) -> list[str]:
    """Parse '  - Name (...)` lines from live schema string."""
    out: list[str] = []
    for line in (schema_context or "").splitlines():
        m = re.match(r"^\s*-\s+(.+?)\s+\(", line)
        if m:
            out.append(m.group(1).strip())
    return out


def fuzzy_match_neighborhood(user_input: str, valid_names: list[str]) -> dict[str, Any]:
    """
    Fuzzy match user neighborhood input against valid DB names.
    Returns best match and confidence score in [0,1].
    """

    def normalize(s: str) -> str:
        s = unicodedata.normalize("NFD", (s or "").lower().strip())
        return "".join(c for c in s if unicodedata.category(c) != "Mn")

    if not user_input or not valid_names:
        return {"name": None, "score": 0.0}
    user_norm = normalize(user_input)
    best_score = 0.0
    best_name: str | None = None
    for name in valid_names:
        name_norm = normalize(name)
        score = SequenceMatcher(None, user_norm, name_norm).ratio()
        if user_norm and (user_norm in name_norm or name_norm in user_norm):
            score = max(score, 0.82)
        if score > best_score:
            best_score = score
            best_name = name
    return {"name": best_name, "score": best_score}


# One replan attempt after validation or output-completeness issues (total 2 planner calls max).
MAX_OUTPUT_CORRECTION_ATTEMPTS = 1


def _normalize_follow_ups_payload(raw: Any) -> list[str]:
    """Turn JSON array, object with suggestions, or list of objects into pill strings."""
    out: list[str] = []

    def _append_one(s: str | None) -> None:
        if not s or not str(s).strip():
            return
        t = str(s).strip()
        words = t.split()
        if len(words) > 12:
            t = " ".join(words[:12])
        out.append(t)

    if raw is None:
        return out
    if isinstance(raw, str):
        _append_one(raw)
        return out[:3]
    if isinstance(raw, dict):
        if "suggestions" in raw and isinstance(raw["suggestions"], list):
            return _normalize_follow_ups_payload(raw["suggestions"])
        for k in ("follow_ups", "pills", "actions"):
            if k in raw and isinstance(raw[k], list):
                return _normalize_follow_ups_payload(raw[k])
        if "label" in raw or "text" in raw:
            _append_one(
                raw.get("label")
                or raw.get("text")
                or raw.get("title")
            )
            return out[:3]
        return out
    if isinstance(raw, list):
        for x in raw:
            if isinstance(x, str):
                _append_one(x)
            elif isinstance(x, dict):
                lab = (
                    x.get("label")
                    or x.get("text")
                    or x.get("title")
                    or x.get("action")
                )
                if isinstance(lab, str) and lab.strip():
                    _append_one(lab)
                elif x.get("label") is not None:
                    _append_one(str(x.get("label")))
            else:
                _append_one(str(x))
            if len(out) >= 3:
                break
        return out
    return out


def strip_follow_ups_suffix(text: str) -> tuple[str, list[str]]:
    """
    Remove <!-- FOLLOW_UPS: ... --> from synthesiser output (array or JSON object); return prose + pills.
    """
    if not (text or "").strip():
        return "", []
    t = text.strip()
    m_open = re.search(r"<!--\s*FOLLOW_UPS:\s*", t, re.IGNORECASE)
    if not m_open:
        return t, []
    start = m_open.start()
    payload_start = m_open.end()
    close = t.find("-->", payload_start)
    if close == -1:
        return t, []
    payload = t[payload_start:close].strip()
    prose = t[:start].strip()
    try:
        raw = json.loads(payload)
    except json.JSONDecodeError:
        return prose, []
    labels = _normalize_follow_ups_payload(raw)
    return prose, labels[:3]


def _dedup_params_signature(params: dict[str, Any]) -> str:
    """Stable fingerprint for render deduplication (excludes display-only keys)."""
    skip = frozenset({"output_intent", "chart_style"})
    d = {k: v for k, v in (params or {}).items() if k not in skip}
    try:
        return json.dumps(d, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return str(d)


_VALID_OUTPUT_INTENTS = frozenset(
    {
        "auto",
        "text",
        "table",
        "cards",
        "metrics",
        "ranking",
        "map",
        "map_listings",
        "map_neighborhoods",
        "combined_map",
        "chart",
        "bar_chart",
        "transit_map",
        "tourism_map",
    }
)


def normalize_output_intent(value: Any) -> str:
    """Normalize the model's display commitment while preserving v1 aliases."""
    oi = str(value or "auto").strip().lower()
    aliases = {
        "listing_map": "map_listings",
        "listings_map": "map_listings",
        "point_map": "map_listings",
        "neighborhood_map": "map_neighborhoods",
        "neighborhoods_map": "map_neighborhoods",
        "barrio_map": "map_neighborhoods",
        "bar": "bar_chart",
        "scatter": "chart",
        "kpi": "metrics",
        "metric_cards": "cards",
    }
    oi = aliases.get(oi, oi)
    return oi if oi in _VALID_OUTPUT_INTENTS else "auto"


def normalize_output_intent_for_tool(tool: str, value: Any) -> str:
    """Clamp broad display intents to values each current tool can satisfy."""
    oi = normalize_output_intent(value)
    if oi == "auto":
        return oi
    if tool == "query_listings":
        if oi == "map":
            return "map_listings"
        if oi in {"map_listings", "cards", "metrics", "table", "ranking", "combined_map"}:
            return oi
        return "auto"
    if tool == "query_neighborhood_profile":
        if oi == "map":
            return "map_neighborhoods"
        if oi in {
            "map_neighborhoods",
            "bar_chart",
            "chart",
            "cards",
            "metrics",
            "table",
            "ranking",
        }:
            return oi
        return "auto"
    if tool == "query_transit_stops":
        if oi == "map":
            return "transit_map"
        return oi if oi in {"transit_map", "combined_map", "table"} else "auto"
    if tool == "query_tourist_apartments":
        if oi == "map":
            return "tourism_map"
        return oi if oi in {"tourism_map", "combined_map", "table"} else "auto"
    if tool == "query_price_trends":
        return oi if oi in {"table", "bar_chart", "chart"} else "auto"
    if tool == "query_chart_data":
        return "chart" if oi in {"chart", "bar_chart"} else "auto"
    if tool == "query_parcel_metrics":
        return oi if oi in {"table", "bar_chart", "chart", "metrics"} else "auto"
    if tool == "compare_neighborhoods":
        return oi if oi in {"table", "bar_chart", "chart", "ranking"} else "auto"
    if tool == "query_neighborhood_density_chart":
        return oi if oi in {"bar_chart", "chart", "table"} else "auto"
    if tool in {"resolve_spatial_reference", "query_neighborhood_context"}:
        return oi if oi in {"table", "text"} else "auto"
    return oi


def _dedupe_render_blocks(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop consecutive duplicate intents with same tool + param fingerprint."""
    seen: set[tuple[str, str, str]] = set()
    out: list[dict[str, Any]] = []
    for b in blocks:
        intent = b.get("intent") or ""
        meta = dict(b.get("meta") or {})
        if intent == "combined_map":
            out.append(b)
            continue
        tool = (meta.get("tool") or "").strip()
        dps = (meta.get("dedup_params") or "").strip()
        key = (intent, tool, dps)
        if key in seen:
            _LOG.warning(
                "Dropping duplicate render block: intent=%s tool=%s params=%s",
                intent,
                tool,
                (dps[:160] + "…") if len(dps) > 160 else dps,
            )
            continue
        seen.add(key)
        out.append(b)
    return out


# User asked for a visual chart; profile rows are present but planner chose a tabular intent.
_CHART_REQUEST_RE = re.compile(
    r"(?:^|\s|[\W_])"
    r"(?:gráfica|gráfico|grafica|grafico|gráficas|gráficos|chart|charts|graph|graphs|plot|plots|scatter)"
    r"(?:$|\s|[\W_])",
    re.IGNORECASE,
)


def validate_output_completeness(
    question: str,
    output_intent: str,
    execution_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Light checks for replan: declared output intent vs result shape, required columns.
    `output_intent` is the v1 display contract until RenderPlan exists.
    """
    issues: list[dict[str, Any]] = []
    q = (question or "").strip()
    plan_oi = normalize_output_intent(output_intent)

    for result in execution_results or []:
        if not result.get("success"):
            continue
        rows = result.get("rows") or []
        tool = str(result.get("tool") or "")
        result_oi = normalize_output_intent_for_tool(
            tool, result.get("output_intent") or plan_oi
        )

        if not rows:
            continue

        columns = set(rows[0].keys())
        params = result.get("params") if isinstance(result.get("params"), dict) else {}
        cstyle = (params.get("chart_style") or "auto").strip().lower()
        if cstyle not in ("bar", "scatter", "auto"):
            cstyle = "auto"
        if (
            tool == "query_neighborhood_profile"
            and len(rows) >= 2
            and cstyle == "auto"
            and _CHART_REQUEST_RE.search(q)
        ):
            issues.append(
                {
                    "type": "chart_intent_mismatch",
                    "tool": tool,
                    "columns_present": sorted(columns),
                    "fix": (
                        "The user asked for a chart. Set **chart_style** to bar (ranking) or "
                        "scatter (yield vs score) on query_neighborhood_profile."
                    ),
                }
            )

        if result_oi in {"map_listings", "transit_map", "tourism_map"}:
            has_coords = "lat" in columns and "lng" in columns
            if not has_coords:
                issues.append(
                    {
                        "type": "missing_map_coordinates",
                        "tool": tool,
                        "columns_present": sorted(columns),
                        "fix": (
                            "The requested output intent is a point map. Include lat and lng "
                            "columns, or choose a non-map output_intent."
                        ),
                    }
                )

        if result_oi == "map_neighborhoods" and tool == "query_neighborhood_profile":
            has_name = any(
                col in columns for col in ("neighborhood_name", "name", "barrio")
            )
            if not has_name:
                issues.append(
                    {
                        "type": "missing_neighborhood_for_map",
                        "tool": tool,
                        "columns_present": sorted(columns),
                        "fix": (
                            "For map_neighborhoods, include the barrio label "
                            "(neighborhood_name/name/barrio) so the renderer can match polygons."
                        ),
                    }
                )

        if tool in (
            "query_neighborhood_profile",
            "query_listings",
            "query_transit_stops",
            "query_tourist_apartments",
        ):
            has_name = any(
                col in columns for col in ("neighborhood_name", "name", "barrio")
            )
            if not has_name:
                issues.append(
                    {
                        "type": "missing_neighborhood_name",
                        "tool": tool,
                        "columns_present": sorted(columns),
                        "fix": (
                            "Include n.name AS neighborhood_name by joining core.neighborhoods "
                            "when filtering by barrio. Never omit the neighborhood label column."
                        ),
                    }
                )

        if tool == "query_listings" and "url" not in columns:
            issues.append(
                {
                    "type": "missing_url",
                    "tool": tool,
                    "columns_present": sorted(columns),
                    "fix": (
                        "Always include l.url in listing queries so users can open Idealista."
                    ),
                }
            )

    return issues


def format_output_completeness_correction(
    issues: list[dict[str, Any]],
    question: str,
    output_intent: str,
) -> str:
    """Build user message block for a replan after output completeness issues."""
    lines: list[str] = []
    for issue in issues:
        lines.append(f"- {issue.get('type', '?')}: {issue.get('fix', '')}")
        lines.append(f"  Columns returned: {issue.get('columns_present', [])}")
    body = "\n".join(lines)
    return f"""YOUR PREVIOUS QUERY HAD THESE PROBLEMS — FIX THEM:
{body}

The user asked: "{question}"
Output intent was: "{normalize_output_intent(output_intent)}"

Generate new tool_calls that fix all problems listed above.
Do not repeat the same mistake."""


def format_validation_plan_correction(
    errors: list[str],
    raw_tool_calls: list[dict[str, Any]],
) -> str:
    """Build user message block for a replan after validate_plan dropped all tool calls."""
    err_lines = "\n".join(f"- {e}" for e in (errors or []) if str(e).strip()) or "(no details)"
    tools_tried = ", ".join(
        str(c.get("tool") or "?") for c in (raw_tool_calls or []) if isinstance(c, dict)
    ) or "(none)"
    return f"""YOUR PREVIOUS TOOL CALLS FAILED VALIDATION — FIX THEM:

Validation errors:
{err_lines}

Tools you attempted: {tools_tried}

Rules:
- For neighborhood filters, use names from the LIVE DATABASE STATE list (fuzzy match in params).
- Prefer **compare_neighborhoods** over multiple profile calls when comparing several barrios.

Generate new tool_calls that satisfy all validation rules. Do not repeat the same mistakes."""


def validate_plan(plan: dict[str, Any], schema_context: str) -> dict[str, Any]:
    """Validate and correct planner output; pure Python."""
    valid_names = extract_neighborhood_names_from_schema(schema_context)
    errors: list[str] = []
    corrected: list[dict[str, Any]] = []
    required_params = {
        "compare_neighborhoods": ["neighborhoods"],
        "resolve_spatial_reference": ["reference"],
        "query_neighborhood_context": ["neighborhood"],
    }
    valid_renderers = {
        "query_listings": [
            "table",
            "point_map",
            "combined_map",
            "metric_cards",
            "no_coords_fallback",
        ],
        "query_transit_stops": ["transit_map", "combined_map"],
        "query_tourist_apartments": ["tourism_map", "combined_map"],
        "query_neighborhood_profile": [
            "bar_chart",
            "metric_cards",
            "table",
            "neighborhood_highlight_map",
            "profile_scatter",
        ],
        "query_price_trends": ["bar_chart", "table"],
        "query_chart_data": ["chart"],
        "query_parcel_metrics": ["table", "bar_chart", "search"],
        "compare_neighborhoods": ["table", "bar_chart", "profile_scatter"],
        "resolve_spatial_reference": ["search", "table"],
        "query_neighborhood_density_chart": ["bar_chart", "chart", "table"],
        "query_neighborhood_context": ["search", "table"],
    }

    for call in plan.get("tool_calls") or []:
        if not isinstance(call, dict):
            continue
        tool = call.get("tool")
        params = dict(call.get("params") or {})
        if not isinstance(tool, str) or not tool:
            errors.append("Invalid tool call")
            continue
        params["output_intent"] = normalize_output_intent_for_tool(
            tool, params.get("output_intent") or plan.get("output_intent")
        )

        if tool == "query_chart_data":
            ct = (params.get("chart_type") or "scatter").strip().lower()
            if ct not in ("scatter", "amenity", "floor"):
                ct = "scatter"
            params["chart_type"] = ct

        nb = params.get("neighborhood")
        if tool not in (
            "query_chart_data",
            "compare_neighborhoods",
            "query_parcel_metrics",
            "query_neighborhood_density_chart",
        ) and isinstance(nb, str) and nb.strip():
            fm = fuzzy_match_neighborhood(nb.strip(), valid_names)
            if fm["score"] < 0.28 and valid_names:
                errors.append(f"Neighborhood '{nb}' not found in live list (best score {fm['score']:.2f})")
                continue
            if fm["name"] and fm["name"] != nb.strip():
                params["neighborhood"] = fm["name"]

        if tool == "compare_neighborhoods":
            nbs = params.get("neighborhoods")
            if not isinstance(nbs, list) or not nbs:
                errors.append("compare_neighborhoods: neighborhoods list required")
                continue
            fixed: list[str] = []
            for raw in nbs:
                if not isinstance(raw, str) or not raw.strip():
                    continue
                fm = fuzzy_match_neighborhood(raw.strip(), valid_names)
                if fm["name"]:
                    fixed.append(fm["name"])
            if not fixed:
                errors.append("compare_neighborhoods: no valid neighborhood names in list")
                continue
            params["neighborhoods"] = fixed

        if tool == "query_parcel_metrics":
            nbs = params.get("neighborhoods")
            if isinstance(nbs, list) and nbs:
                fixed2: list[str] = []
                for raw in nbs:
                    if not isinstance(raw, str) or not raw.strip():
                        continue
                    fm = fuzzy_match_neighborhood(raw.strip(), valid_names)
                    if fm["name"]:
                        fixed2.append(fm["name"])
                params["neighborhoods"] = fixed2

        if tool == "query_neighborhood_density_chart":
            nbs = params.get("neighborhoods")
            if isinstance(nbs, list) and nbs:
                fixed3: list[str] = []
                for raw in nbs:
                    if not isinstance(raw, str) or not raw.strip():
                        continue
                    fm = fuzzy_match_neighborhood(raw.strip(), valid_names)
                    if fm["name"]:
                        fixed3.append(fm["name"])
                params["neighborhoods"] = fixed3
            m = (params.get("metric") or "tourist_density_pct").strip().lower()
            if m not in ("tourist_density_pct", "tourist_apt_count"):
                m = "tourist_density_pct"
            params["metric"] = m

        if tool == "query_neighborhood_context" and isinstance(
            params.get("neighborhood"), str
        ):
            fnb = (params.get("neighborhood") or "").strip()
            if fnb and valid_names:
                fm2 = fuzzy_match_neighborhood(fnb, valid_names)
                if fm2["name"]:
                    params["neighborhood"] = fm2["name"]
                else:
                    errors.append(f"Neighborhood not found: {fnb!r}")
                    continue

        req = required_params.get(tool, [])
        missing = [r for r in req if not params.get(r)]
        if missing:
            errors.append(f"Tool {tool} missing required params: {missing}")
            continue

        vr = valid_renderers.get(tool, ["table"])
        ren = call.get("renderer") or "table"
        if ren not in vr:
            ren = vr[0]
        corrected.append({**call, "params": params, "renderer": ren})

    out = dict(plan)
    out["tool_calls"] = corrected
    out["validation_errors"] = errors
    out["output_intent"] = normalize_output_intent(plan.get("output_intent"))
    if not plan.get("tool_calls"):
        out["valid"] = len(errors) == 0
    else:
        out["valid"] = len(errors) == 0 and len(corrected) > 0
    return out


def _user_wants_at_least_rooms_not_exact(user_message: str) -> bool:
    """
    True when the user text suggests a *floor* on bedrooms, not an exact count.
    If False and they mention rooms, 'N habitaciones' is treated as exact elsewhere.
    """
    t = (user_message or "").lower()
    if any(
        p in t
        for p in (
            "al menos",
            "almenos",
            "mínimo",
            "minimo",
            "minimum",
            "at least",
            "más de ",
            "mas de ",
            "como mínimo",
            "como minimo",
            "upwards of",
        )
    ):
        return True
    if "entre" in t and "habit" in t:
        return True
    if re.search(r"\d+\s+o\s+más", t) or re.search(r"\d+\s+o\s+mas", t):
        return True
    return False


def _user_message_mentions_room_count(user_message: str) -> bool:
    t = (user_message or "").lower()
    return bool(
        re.search(
            r"\b(?:habitaciones|hab\.|dormitorios|dorm\.?)\b",
            t,
            re.IGNORECASE,
        )
    )


def _normalize_query_listings_room_params(
    params: dict[str, Any], user_message: str | None
) -> dict[str, Any]:
    """
    If the model only passes min_rooms for an exact-N query, SQL would use >= N and
    return larger units. When the user message looks like an exact room count (not
    'at least'), set max_rooms = min_rooms.
    """
    mr = params.get("min_rooms")
    xr = params.get("max_rooms")
    if mr is None or xr is not None:
        return params
    if not user_message or not _user_message_mentions_room_count(user_message):
        return params
    if _user_wants_at_least_rooms_not_exact(user_message):
        return params
    try:
        n = int(mr)
    except (TypeError, ValueError):
        return params
    return {**params, "max_rooms": n}


def query_listings_fn(params: dict[str, Any], engine) -> list[dict[str, Any]]:
    conditions = ["l.price_int > 0", "l.area_sqm > 0"]
    nb = params.get("neighborhood")
    if isinstance(nb, str) and nb.strip():
        es = _sql_escape(nb.strip())
        conditions.append(
            f"similarity(unaccent(lower(n.name)), unaccent(lower('{es}'))) > 0.4"
        )
    op = (params.get("operation") or "venta").strip().lower()
    if op in ("venta", "alquiler"):
        conditions.append(f"l.operation = '{op}'")
    if params.get("max_price") is not None:
        try:
            conditions.append(f"l.price_int <= {int(params['max_price'])}")
        except (TypeError, ValueError):
            pass
    if params.get("min_price") is not None:
        try:
            conditions.append(f"l.price_int >= {int(params['min_price'])}")
        except (TypeError, ValueError):
            pass
    if params.get("min_rooms") is not None:
        try:
            conditions.append(f"l.rooms_int >= {int(params['min_rooms'])}")
        except (TypeError, ValueError):
            pass
    if params.get("max_rooms") is not None:
        try:
            conditions.append(f"l.rooms_int <= {int(params['max_rooms'])}")
        except (TypeError, ValueError):
            pass
    if params.get("only_below_median"):
        conditions.append(
            """(
            (l.operation = 'venta' AND l.price_int < np.median_venta_price)
            OR (l.operation = 'alquiler' AND l.price_int < np.median_alquiler_price)
        )"""
        )
    amenity_map = {
        "parking": "l.has_parking",
        "terrace": "l.has_terrace",
        "elevator": "l.has_elevator",
        "ac": "l.has_ac",
        "renovated": "l.is_renovated",
    }
    for a in params.get("amenities") or []:
        if isinstance(a, str) and a.lower() in amenity_map:
            conditions.append(f"{amenity_map[a.lower()]} = true")
    lim = 25
    try:
        lim = max(1, min(100, int(params.get("limit", 25))))
    except (TypeError, ValueError):
        lim = 25
    where_clause = " AND ".join(conditions)
    sql = f"""
        SELECT
            l.url,
            l.operation,
            l.price_int,
            l.area_sqm,
            l.rooms_int,
            ROUND(l.floor_int)::integer AS floor_int,
            ROUND((l.price_int::numeric / NULLIF(l.area_sqm::numeric, 0)), 0) AS eur_per_sqm,
            l.has_parking, l.has_terrace, l.has_elevator, l.is_renovated,
            l.lat, l.lng, l.geocode_quality,
            n.name AS neighborhood_name,
            np.gross_rental_yield_pct AS neighborhood_yield,
            np.investment_score,
            np.median_venta_price AS neighborhood_median,
            CASE WHEN l.operation = 'venta' AND l.price_int < np.median_venta_price THEN true
                 WHEN l.operation = 'alquiler' AND l.price_int < np.median_alquiler_price THEN true
                 ELSE false END AS below_median
        FROM core.listings l
        JOIN core.neighborhoods n ON n.id = l.neighborhood_id
        LEFT JOIN analytics.neighborhood_profile np ON np.neighborhood_id = l.neighborhood_id
        WHERE {where_clause}
        ORDER BY l.price_int ASC NULLS LAST
        LIMIT {lim}
    """
    df = pd.read_sql(text(sql), engine)
    return df.to_dict("records")


def query_neighborhood_profile_fn(params: dict[str, Any], engine) -> list[dict[str, Any]]:
    order_by_map = {
        "investment_score": "np.investment_score",
        "yield": "np.gross_rental_yield_pct",
        "price": "np.median_venta_eur_per_sqm",
        "listings": "np.total_count",
    }
    order_col = order_by_map.get(
        (params.get("order_by") or "investment_score").strip().lower(),
        "np.investment_score",
    )
    try:
        min_listings = max(0, int(params.get("min_listings", 3)))
    except (TypeError, ValueError):
        min_listings = 3
    neighborhoods = params.get("neighborhoods") or []
    nb_filter = ""
    if isinstance(neighborhoods, list) and neighborhoods:
        parts = []
        for nm in neighborhoods[:12]:
            if not isinstance(nm, str):
                continue
            es = _sql_escape(nm.strip())
            parts.append(
                f"similarity(unaccent(lower(n.name)), unaccent(lower('{es}'))) > 0.4"
            )
        if parts:
            nb_filter = "AND (" + " OR ".join(parts) + ")"

    sql = f"""
        SELECT
            n.name AS neighborhood_name,
            np.gross_rental_yield_pct AS yield_pct,
            np.investment_score AS value,
            np.median_venta_eur_per_sqm AS eur_per_sqm,
            np.median_alquiler_price AS median_rent,
            np.median_venta_price AS median_sale,
            np.transport_rating,
            np.tourist_density_pct AS tourist_pct,
            np.tourism_pressure,
            np.transit_stop_count,
            np.venta_count,
            np.alquiler_count,
            np.total_count
        FROM analytics.neighborhood_profile np
        JOIN core.neighborhoods n ON n.id = np.neighborhood_id
        WHERE np.total_count >= {min_listings}
          AND np.gross_rental_yield_pct IS NOT NULL
          {nb_filter}
        ORDER BY {order_col} DESC NULLS LAST
        LIMIT 15
    """
    df = pd.read_sql(text(sql), engine)
    return df.to_dict("records")


def query_transit_stops_fn(params: dict[str, Any], engine) -> list[dict[str, Any]]:
    nb = params.get("neighborhood")
    if isinstance(nb, str) and nb.strip():
        es = _sql_escape(nb.strip())
        where = f"WHERE similarity(unaccent(lower(n.name)), unaccent(lower('{es}'))) > 0.4"
    else:
        where = "WHERE TRUE"
    sql = f"""
        SELECT t.name, t.stop_type, t.lat, t.lng, n.name AS neighborhood_name
        FROM core.transit_stops t
        JOIN core.neighborhoods n ON n.id = t.neighborhood_id
        {where}
        ORDER BY t.stop_type NULLS LAST
        LIMIT 2000
    """
    df = pd.read_sql(text(sql), engine)
    return df.to_dict("records")


def query_tourist_apartments_fn(params: dict[str, Any], engine) -> list[dict[str, Any]]:
    nb = params.get("neighborhood")
    if isinstance(nb, str) and nb.strip():
        es = _sql_escape(nb.strip())
        where = f"""WHERE similarity(unaccent(lower(n.name)), unaccent(lower('{es}'))) > 0.4
          AND (ta.status IS NULL OR lower(ta.status) = 'active')"""
    else:
        where = "WHERE (ta.status IS NULL OR lower(ta.status) = 'active')"
    sql = f"""
        SELECT ta.id, ta.address, ta.lat, ta.lng, n.name AS neighborhood_name
        FROM core.tourist_apartments ta
        JOIN core.neighborhoods n ON n.id = ta.neighborhood_id
        {where}
        LIMIT 2000
    """
    df = pd.read_sql(text(sql), engine)
    return df.to_dict("records")


def query_price_trends_fn(params: dict[str, Any], engine) -> list[dict[str, Any]]:
    nb = params.get("neighborhood")
    if isinstance(nb, str) and nb.strip():
        es = _sql_escape(nb.strip())
        nb_filter = f"AND similarity(unaccent(lower(n.name)), unaccent(lower('{es}'))) > 0.4"
    else:
        nb_filter = ""
    sql = f"""
        SELECT
            pc.url,
            n.name AS neighborhood_name,
            pc.price_int,
            pc.price_int_previous,
            pc.price_drop_eur,
            pc.price_drop_pct,
            l.lat,
            l.lng
        FROM analytics.price_changes pc
        JOIN core.listings l ON l.url = pc.url
        JOIN core.neighborhoods n ON n.id = l.neighborhood_id
        WHERE l.price_int > 0
          {nb_filter}
        ORDER BY ABS(pc.price_drop_pct) DESC NULLS LAST
        LIMIT 50
    """
    df = pd.read_sql(text(sql), engine)
    return df.to_dict("records")


def query_chart_data_fn(params: dict[str, Any], engine) -> list[dict[str, Any]]:
    """Plotly charts load from listings snapshot in renderers; executor only signals success."""
    del params, engine
    return [{"_chart": True}]


def query_parcel_metrics_fn(params: dict[str, Any], engine) -> list[dict[str, Any]]:
    nbs = params.get("neighborhoods") or []
    if nbs and isinstance(nbs, list):
        parts = []
        for x in nbs:
            if isinstance(x, str) and x.strip():
                parts.append("'" + _sql_escape(x.strip()) + "'")
        where = f"WHERE neighborhood_name IN ({','.join(parts)})" if parts else ""
    else:
        where = ""
    sql = f"""
        SELECT * FROM analytics.parcel_metrics
        {where}
        ORDER BY parcel_count DESC NULLS LAST
        LIMIT 30
    """
    df = pd.read_sql(text(sql), engine)
    return df.to_dict("records")


def compare_neighborhoods_fn(params: dict[str, Any], engine) -> list[dict[str, Any]]:
    nbs = [x for x in (params.get("neighborhoods") or []) if isinstance(x, str) and x.strip()]
    if not nbs:
        return []
    parts: list[str] = []
    for x in nbs:
        parts.append("'" + _sql_escape(x.strip()) + "'")
    sql = f"""
        SELECT
            neighborhood_name,
            gross_rental_yield_pct,
            investment_score,
            tourist_density_pct,
            transit_stop_count,
            avg_dist_to_stop_m,
            total_count,
            median_venta_price,
            median_alquiler_price
        FROM analytics.neighborhood_profile
        WHERE neighborhood_name IN ({",".join(parts)})
        ORDER BY investment_score DESC NULLS LAST
    """
    df = pd.read_sql(text(sql), engine)
    return df.to_dict("records")


def resolve_spatial_reference_fn(params: dict[str, Any], engine) -> list[dict[str, Any]]:
    from agent.spatial_resolver import match_reference_phrase

    rows = match_reference_phrase(str(params.get("reference") or ""))
    names = [r.get("name") for r in rows if isinstance(r.get("name"), str)]
    if not names:
        return rows
    escaped = ",".join("'" + _sql_escape(str(n)) + "'" for n in names)
    sql = f"""
        SELECT id::text AS neighborhood_id, name AS neighborhood_name
        FROM core.neighborhoods
        WHERE name IN ({escaped})
    """
    try:
        df = pd.read_sql(text(sql), engine)
    except Exception:
        return rows
    id_by_name = {
        str(r["neighborhood_name"]): str(r["neighborhood_id"])
        for r in df.to_dict("records")
    }
    out: list[dict[str, Any]] = []
    for r in rows:
        n = str(r.get("name") or "")
        out.append(
            {
                "neighborhood_id": id_by_name.get(n),
                "neighborhood_name": n,
                "confidence": r.get("confidence", 0.0),
                "source": "spatial_lexicon",
            }
        )
    return out


def query_neighborhood_density_chart_fn(
    params: dict[str, Any], engine
) -> list[dict[str, Any]]:
    nbs = [x for x in (params.get("neighborhoods") or []) if isinstance(x, str) and x.strip()]
    metric = (params.get("metric") or "tourist_density_pct").strip()
    mcol = "tourist_apt_count" if metric == "tourist_apt_count" else "tourist_density_pct"
    if nbs:
        parts = ["'" + _sql_escape(x.strip()) + "'" for x in nbs]
        where = f"WHERE np.neighborhood_name IN ({','.join(parts)})"
    else:
        where = f"WHERE np.{mcol} IS NOT NULL"
    order = f"np.{mcol} DESC NULLS LAST"
    sql = f"""
        SELECT np.neighborhood_name, np.{mcol} AS value
        FROM analytics.neighborhood_profile np
        {where}
        ORDER BY {order}
        LIMIT 25
    """
    df = pd.read_sql(text(sql), engine)
    return df.to_dict("records")


def query_neighborhood_context_fn(params: dict[str, Any], engine) -> list[dict[str, Any]]:
    del engine, params
    return []


TOOL_FUNCTIONS: dict[str, Any] = {
    "query_listings": query_listings_fn,
    "query_neighborhood_profile": query_neighborhood_profile_fn,
    "query_transit_stops": query_transit_stops_fn,
    "query_tourist_apartments": query_tourist_apartments_fn,
    "query_price_trends": query_price_trends_fn,
    "query_chart_data": query_chart_data_fn,
    "query_parcel_metrics": query_parcel_metrics_fn,
    "compare_neighborhoods": compare_neighborhoods_fn,
    "resolve_spatial_reference": resolve_spatial_reference_fn,
    "query_neighborhood_density_chart": query_neighborhood_density_chart_fn,
    "query_neighborhood_context": query_neighborhood_context_fn,
}


def execute_plan(
    validated_plan: dict[str, Any],
    engine,
    user_message: str | None = None,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for call in validated_plan.get("tool_calls") or []:
        tool = call.get("tool")
        params = dict(call.get("params") or {})
        oi_call = normalize_output_intent_for_tool(
            tool if isinstance(tool, str) else "",
            params.get("output_intent") or validated_plan.get("output_intent"),
        )
        params["output_intent"] = oi_call
        _sql_meta_keys = frozenset({"output_intent", "chart_style"})
        sql_params = {k: v for k, v in params.items() if k not in _sql_meta_keys}
        if tool == "query_listings":
            sql_params = _normalize_query_listings_room_params(sql_params, user_message)
        renderer = call.get("renderer") or "table"
        tool_call_id = call.get("_tool_call_id") or call.get("tool_call_id")
        if not isinstance(tool, str) or tool not in TOOL_FUNCTIONS:
            results.append(
                {
                    "tool": tool,
                    "params": params,
                    "renderer": renderer,
                    "rows": [],
                    "row_count": 0,
                    "success": False,
                    "error": f"Unknown tool: {tool}",
                    "tool_call_id": tool_call_id,
                    "output_intent": oi_call,
                }
            )
            continue
        try:
            rows = TOOL_FUNCTIONS[tool](sql_params, engine)
            results.append(
                {
                    "tool": tool,
                    "params": params,
                    "renderer": renderer,
                    "rows": rows,
                    "row_count": len(rows),
                    "success": True,
                    "error": None,
                    "tool_call_id": tool_call_id,
                    "output_intent": oi_call,
                }
            )
        except Exception as e:
            results.append(
                {
                    "tool": tool,
                    "params": params,
                    "renderer": renderer,
                    "rows": [],
                    "row_count": 0,
                    "success": False,
                    "error": str(e),
                    "tool_call_id": tool_call_id,
                    "output_intent": oi_call,
                }
            )
    return results


def decide_renderer(result: dict[str, Any], output_intent: str = "auto") -> str:
    """Pick Streamlit renderer from output_intent first; fallback to data-shape heuristics."""
    tool = result.get("tool")
    oi = normalize_output_intent_for_tool(
        tool if isinstance(tool, str) else "",
        output_intent,
    )
    rows = result.get("rows") or []
    planned = result.get("renderer") or "table"
    row_count = len(rows)

    def _has_coords() -> bool:
        return any(
            r.get("lat") is not None and r.get("lng") is not None for r in rows
        )

    if tool == "query_chart_data":
        return "chart" if row_count > 0 else "empty"
    if row_count == 0:
        return "empty"

    if tool == "query_neighborhood_profile" and row_count >= 2:
        p = result.get("params") or {}
        cstyle = (p.get("chart_style") or "auto").strip().lower()
        if cstyle not in ("bar", "scatter", "auto"):
            cstyle = "auto"
        if cstyle == "scatter":
            return "profile_scatter"
        if cstyle == "bar":
            return "metric_cards" if row_count == 1 else "bar_chart"

    if oi != "auto":
        if oi == "text":
            return "empty"
        if oi == "bar_chart":
            if tool == "query_neighborhood_profile":
                return "metric_cards" if row_count == 1 else "bar_chart"
            if tool == "query_price_trends":
                return "table" if row_count > 8 else "bar_chart"
            if tool == "query_chart_data":
                return "chart"
            if tool in {"compare_neighborhoods", "query_neighborhood_density_chart"}:
                return "bar_chart"
            return planned
        if oi == "transit_map":
            if tool == "query_transit_stops":
                return "transit_map"
            return planned
        if oi == "tourism_map":
            if tool == "query_tourist_apartments":
                return "tourism_map"
            return planned
        if oi == "map_listings":
            if tool == "query_listings":
                return "point_map" if _has_coords() else "no_coords_fallback"
            return planned
        if oi == "map_neighborhoods":
            if tool == "query_neighborhood_profile":
                return "neighborhood_highlight_map"
            return planned
        if oi == "cards":
            if tool == "query_neighborhood_profile":
                return "metric_cards" if row_count <= 6 else "bar_chart"
            if tool == "query_listings":
                return "metric_cards" if row_count <= 3 else "table"
            if tool == "query_price_trends":
                return "table"
            return "metric_cards" if row_count <= 3 else "table"
        if oi == "map":
            if tool == "query_listings":
                return "point_map" if _has_coords() else "table"
            if tool == "query_transit_stops":
                return "transit_map"
            if tool == "query_tourist_apartments":
                return "tourism_map"
            if tool == "query_neighborhood_profile":
                return "neighborhood_highlight_map"
            return planned
        if oi == "chart":
            if tool == "query_chart_data":
                return "chart"
            if tool == "query_neighborhood_profile":
                return "metric_cards" if row_count == 1 else "bar_chart"
            if tool in {"compare_neighborhoods", "query_neighborhood_density_chart"}:
                return "bar_chart"
            return planned
        if oi == "metrics":
            if tool == "query_neighborhood_profile":
                return "metric_cards" if row_count <= 6 else "bar_chart"
            if tool == "query_listings":
                return "metric_cards" if row_count <= 3 else "table"
            if tool == "query_price_trends":
                return "table"
            if tool == "query_parcel_metrics":
                return "table"
            return planned
        if oi == "table":
            if tool == "query_listings":
                return "metric_cards" if row_count <= 3 else "table"
            if tool == "query_neighborhood_profile":
                return (
                    "metric_cards"
                    if row_count == 1
                    else ("bar_chart" if row_count <= 6 else "table")
                )
            if tool == "query_price_trends":
                return "table"
            return "table"
        if oi == "ranking":
            if tool == "query_neighborhood_profile":
                return "bar_chart"
            if tool in {"query_listings", "query_price_trends", "compare_neighborhoods"}:
                return "table"
            return planned
        if oi == "combined_map":
            return planned

    if tool == "query_listings":
        has_coords = _has_coords()
        if planned == "point_map" and not has_coords:
            return "table"
        if row_count <= 3 and planned == "table":
            return "metric_cards"
        if row_count > 15 and planned == "metric_cards":
            return "table"
        return planned
    if tool == "query_neighborhood_profile":
        if row_count == 1:
            return "metric_cards"
        if row_count <= 6:
            return "bar_chart"
        return "table"
    if tool == "query_transit_stops":
        return "transit_map"
    if tool == "query_tourist_apartments":
        return "tourism_map"
    if tool == "query_price_trends":
        return "table" if row_count > 8 else planned
    if tool == "query_parcel_metrics":
        return "table"
    if tool == "compare_neighborhoods":
        return "bar_chart" if row_count <= 8 else "table"
    if tool == "resolve_spatial_reference":
        return "search"
    if tool == "query_neighborhood_density_chart":
        return "bar_chart"
    if tool == "query_neighborhood_context":
        return "search"
    return planned


def _llm_synthesiser(
    messages: list[dict[str, str]],
    model: str,
    timeout_sec: float,
    max_tokens: int = 400,
) -> str:
    from agent.responses_api import (
        extract_response_text,
        get_openai_client,
        reasoning_param_for_model,
        supports_temperature,
    )

    system = messages[0]["content"] if messages else SYNTHESISER_SYSTEM_PROMPT
    user_text = "\n\n".join(m.get("content", "") for m in messages[1:] if m.get("content"))
    client = get_openai_client(max(timeout_sec, 35.0))
    kwargs: dict[str, Any] = {
        "model": model,
        "instructions": system,
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": user_text}],
            }
        ],
        "max_output_tokens": max_tokens,
    }
    if supports_temperature(model):
        kwargs["temperature"] = 0.3
    rpar = reasoning_param_for_model(model, "low")
    if rpar is not None:
        kwargs["reasoning"] = rpar
    response = client.responses.create(**kwargs)
    return extract_response_text(response).strip()


def infer_synth_max_tokens(execution_results: list[dict[str, Any]]) -> int:
    """Higher caps for dense visuals (overview cards, profiles, charts)."""
    if not execution_results:
        return 200
    rends = [r.get("renderer") for r in execution_results]
    tools = [r.get("tool") for r in execution_results]
    if "metric_cards" in rends:
        return 300
    if "neighborhood_highlight_map" in rends:
        return 280
    if "profile_scatter" in rends:
        return 280
    if "no_coords_fallback" in rends:
        return 220
    if any(t == "query_neighborhood_profile" for t in tools):
        return 250
    if any(t == "query_chart_data" for t in tools):
        return 200
    return 150


def format_confirmed_visuals(execution_results: list[dict[str, Any]]) -> str:
    """Human-readable list of tools and final renderers for the synthesiser."""
    if not execution_results:
        return "ninguno (solo texto)"
    parts: list[str] = []
    for r in execution_results:
        tool = r.get("tool") or "?"
        ren = (r.get("renderer") or "?").strip()
        if ren == "empty":
            parts.append(f"{tool}→sin filas")
        else:
            parts.append(f"{tool}→{ren}")
    return "; ".join(parts)


def _build_results_summary_for_synth(
    execution_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    results_summary: list[dict[str, Any]] = []
    for result in execution_results:
        tool = result.get("tool")
        if tool == "query_chart_data" and result.get("success"):
            p = result.get("params") or {}
            results_summary.append(
                {
                    "tool": tool,
                    "chart_type": p.get("chart_type") or "scatter",
                    "renderer": result.get("renderer"),
                }
            )
            continue
        if result.get("success") and result.get("row_count", 0) > 0:
            sample = (result.get("rows") or [])[:3]
            results_summary.append(
                {
                    "tool": tool,
                    "row_count": result.get("row_count"),
                    "sample": sample,
                    "renderer": result.get("renderer"),
                }
            )
        elif not result.get("success"):
            results_summary.append({"tool": tool, "error": result.get("error")})
        else:
            results_summary.append(
                {
                    "tool": tool,
                    "row_count": 0,
                    "note": "sin filas",
                }
            )
    return results_summary


def build_synthesiser_messages(
    question: str,
    plan: dict[str, Any],
    execution_results: list[dict[str, Any]],
    conversation_state: dict[str, Any],
    confirmed_visuals: str | None = None,
    max_tokens_override: int | None = None,
) -> tuple[list[dict[str, str]], int]:
    """Messages + max_tokens for the main synthesiser (streaming or not)."""
    results_summary = _build_results_summary_for_synth(execution_results)
    visuals_line = confirmed_visuals or format_confirmed_visuals(execution_results)
    max_tok = infer_synth_max_tokens(execution_results)
    if max_tokens_override is not None:
        max_tok = max_tokens_override
    else:
        max_tok += 50
    user_block = f"""Pregunta del usuario: "{question}"

Plan reasoning: {plan.get("reasoning", "")}

Visuales confirmados (después de tu texto): {visuals_line}
Tu texto debe coincidir con esos visuales (no describas un mapa si solo hay tabla o gráfico).

Resultados de ejecución:
{json.dumps(results_summary, indent=2, default=str)}

Estado de conversación:
{json.dumps(conversation_state, indent=2, default=str)}

Escribe la respuesta en **español** (máx. 3 frases, sin negritas markdown, sin nombres de columnas crudos).
Luego en una línea nueva el comentario HTML <!-- FOLLOW_UPS: [...] --> como en el system prompt."""
    messages = [
        {"role": "system", "content": SYNTHESISER_SYSTEM_PROMPT},
        {"role": "user", "content": user_block},
    ]
    return messages, max_tok


def _resolve_synthesiser_model(model: str | None) -> str:
    if model:
        return model
    return DEFAULT_SYNTHESISER_MODEL_OPENAI


def format_last_assistant_for_planner(messages: list[dict[str, Any]]) -> str:
    """
    Build a compact text block from the last assistant turn so the planner can resolve
    "estos", "them", etc. Uses `summary` and `render_stack` (and legacy row fields) from
    Streamlit chat history.
    """
    for msg in reversed(messages or []):
        if msg.get("role") != "assistant":
            continue
        parts: list[str] = []
        summary = (msg.get("summary") or "").strip()
        if summary:
            parts.append(f"Assistant prose:\n{summary}")

        stack = msg.get("render_stack") or []
        for block in stack:
            intent = (block.get("intent") or "").strip() or "?"
            rows = block.get("rows") or []
            meta = dict(block.get("meta") or {})

            if intent == "combined_map":
                for label, key in (
                    ("listings", "rows_listings"),
                    ("transit", "rows_transit"),
                    ("tourism", "rows_tourism"),
                ):
                    sub = meta.get(key) or []
                    if sub:
                        line = _format_shown_rows_sample(sub, intent=f"combined_map/{label}")
                        if line:
                            parts.append(line)
                continue

            if rows:
                line = _format_shown_rows_sample(rows, intent=intent)
                if line:
                    parts.append(line)

        # Legacy message shape (non-agent stack)
        if msg.get("rows"):
            line = _format_shown_rows_sample(
                msg.get("rows") or [], intent=str(msg.get("intent") or "legacy")
            )
            if line:
                parts.append(line)

        return "\n\n".join(parts) if parts else ""

    return ""


def _format_shown_rows_sample(
    rows: list[dict[str, Any]],
    *,
    intent: str,
    max_rows: int = 8,
) -> str:
    if not rows:
        return ""
    sample = rows[:max_rows]
    first = sample[0]
    if not isinstance(first, dict):
        return ""
    key_field = next(
        (k for k in ("neighborhood_name", "name", "url") if k in first),
        None,
    )
    if not key_field:
        return f"[intent={intent}: {len(rows)} row(s); no neighborhood_name/name/url in first row]"
    shown = [row.get(key_field) for row in sample if row.get(key_field) is not None]
    shown = [str(x) for x in shown if str(x).strip()]
    if not shown:
        return ""
    extra = f" (+{len(rows) - len(shown)} more)" if len(rows) > len(shown) else ""
    return f"[intent={intent}; Showed {key_field}: {shown}{extra}]"


def build_openai_first_turn_messages(
    user_input: str,
    conversation_state: dict[str, Any],
    conversation_context: str,
    live_schema_context: str,
    static_schema: str,
    correction_hint: str | None = None,
    last_assistant_context: str = "",
    resolved_intent: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    """Stack static content first, dynamic context last for prompt-cache hit rate."""
    system = (
        OPENAI_TOOLS_FIRST_SYSTEM_PREFIX
        + "\n\n=== STATIC SCHEMA REFERENCE ===\n"
        + static_schema
        + "\n\n---\n\n=== LIVE DATABASE STATE ===\n"
        + live_schema_context
    )
    correction_section = ""
    if correction_hint:
        correction_section = f"""
=== CORRECTION REQUIRED ===
{correction_hint}
=== END CORRECTION ===

"""
    last_block = (last_assistant_context or "").strip() or (
        "(none — first turn or no prior assistant message in this thread)"
    )
    ri = ""
    if resolved_intent:
        ri = (
            "\n=== RESOLVED INTENT (must respect; use for operation and neighborhoods) ===\n"
            + json.dumps(resolved_intent, indent=2, default=str)
            + "\n"
        )

    user_block = f"""{correction_section}{ri}=== CONVERSATION MEMORY ===
{json.dumps(conversation_state, indent=2, default=str)}

=== LAST ASSISTANT CONTEXT ===
{last_block}

Recent transcript:
{conversation_context or "(none)"}

User question: {user_input}"""
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_block},
    ]


def openai_tool_calls_to_plan_calls(tool_calls: list[Any]) -> list[dict[str, Any]]:
    """Map OpenAI ChatCompletionMessageToolCall objects to Rooster plan tool_calls."""
    out: list[dict[str, Any]] = []
    for tc in tool_calls:
        fn = getattr(tc, "function", None)
        name = getattr(fn, "name", None) if fn else None
        raw_args = getattr(fn, "arguments", None) if fn else None
        if not isinstance(name, str) or not name:
            continue
        try:
            args = json.loads(raw_args or "{}")
        except json.JSONDecodeError:
            args = {}
        if not isinstance(args, dict):
            args = {}
        out.append(
            {
                "tool": name,
                "params": args,
                "renderer": DEFAULT_RENDERER_FOR_TOOL.get(name, "table"),
                "renderer_rationale": "",
                "_tool_call_id": getattr(tc, "id", None),
            }
        )
    return out


def extract_output_intent_from_tool_args(tool_calls: list[Any]) -> str:
    """Return the first valid output_intent emitted by the planner, else auto."""
    for tc in tool_calls or []:
        fn = getattr(tc, "function", None)
        raw_args = getattr(fn, "arguments", None) if fn else None
        try:
            args = json.loads(raw_args or "{}")
        except json.JSONDecodeError:
            args = {}
        if not isinstance(args, dict):
            continue
        name = getattr(fn, "name", "") if fn else ""
        oi = normalize_output_intent_for_tool(str(name or ""), args.get("output_intent"))
        if oi != "auto":
            return oi
    return "auto"


def _infer_plan_neighborhood_resolved(plan: dict[str, Any]) -> None:
    for c in plan.get("tool_calls") or []:
        p = c.get("params") or {}
        nb = p.get("neighborhood")
        if isinstance(nb, str) and nb.strip():
            plan["neighborhood_resolved"] = nb.strip()
            return
        nbs = p.get("neighborhoods")
        if isinstance(nbs, list) and nbs and isinstance(nbs[0], str) and nbs[0].strip():
            plan["neighborhood_resolved"] = nbs[0].strip()
            return
    plan["neighborhood_resolved"] = None


def _infer_combine_maps_from_tools(plan: dict[str, Any]) -> bool:
    spatial = {"query_listings", "query_transit_stops", "query_tourist_apartments"}
    n = sum(1 for c in (plan.get("tool_calls") or []) if c.get("tool") in spatial)
    return n >= 2


def _openai_fc_first_completion(
    messages: list[dict[str, str]],
    model: str,
    timeout_sec: float,
    prompt_cache_key: str | None = None,
    previous_response_id: str | None = None,
) -> Any:
    """Planner first turn via Responses API (tools)."""
    from agent.responses_api import (
        chat_tools_to_responses_tools,
        create_response_with_tools,
        get_openai_client,
    )

    if len(messages) < 2 or messages[0].get("role") != "system":
        raise ValueError("expected [system, user] first-turn messages")
    instructions = str(messages[0].get("content") or "")
    user_input = str(messages[1].get("content") or "")
    client = get_openai_client(max(timeout_sec, 35.0))
    tools = chat_tools_to_responses_tools(get_rooster_openai_tools())
    return create_response_with_tools(
        client,
        model=model,
        instructions=instructions,
        user_input=user_input,
        tools=tools,
        prompt_cache_key=prompt_cache_key,
        previous_response_id=previous_response_id,
    )


def run_openai_function_calling_pipeline(
    user_input: str,
    conversation_state: dict[str, Any],
    live_schema_context: str,
    static_schema: str,
    conversation_context: str,
    model: str,
    timeout_sec: float,
    engine,
    last_assistant_context: str = "",
    prompt_cache_key: str | None = None,
    previous_planner_response_id: str | None = None,
    preamble_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """
    OpenAI native function calling: first response may request tools; we execute, then
    the app runs a follow-up streaming synthesis (Responses API) with tool results in context.
    May replan if output-completeness checks fail (see validate_output_completeness).
    `model` must be the UI-selected OpenAI model id.
    """
    from agent.intent_resolver import resolve_intent

    model_name = _resolve_synthesiser_model(model)
    combined_schema = f"{live_schema_context}\n\n=== STATIC ===\n{static_schema}"
    try:
        _ri = resolve_intent(
            user_input,
            conversation_state,
            combined_schema,
            timeout_sec=min(float(timeout_sec), 25.0),
            prompt_cache_key=prompt_cache_key,
        )
        resolved_intent_dict: dict[str, Any] | None = _ri.model_dump()
    except Exception:
        resolved_intent_dict = None

    correction_hint: str | None = None
    had_output_correction = False
    had_validation_replan = False

    from agent.responses_api import (
        extract_response_text,
        output_items_to_tool_calls_compat,
        tool_json_payloads_to_responses_input,
    )

    last_tcalls: list[Any] = []
    last_validated: dict[str, Any] = {}
    last_execution: list[dict[str, Any]] = []
    last_ve: list[str] = []
    last_planner_response_id: str | None = None
    completed_tool_path: bool = False

    for attempt in range(MAX_OUTPUT_CORRECTION_ATTEMPTS + 1):
        first_messages = build_openai_first_turn_messages(
            user_input,
            conversation_state,
            conversation_context,
            live_schema_context,
            static_schema,
            correction_hint=correction_hint,
            last_assistant_context=last_assistant_context,
            resolved_intent=resolved_intent_dict,
        )

        def _run_first() -> Any:
            return _openai_fc_first_completion(
                first_messages,
                model_name,
                timeout_sec,
                prompt_cache_key=prompt_cache_key,
                previous_response_id=previous_planner_response_id,
            )

        try:
            with ThreadPoolExecutor(max_workers=1) as ex:
                response = ex.submit(_run_first).result(timeout=timeout_sec)
        except FuturesTimeoutError:
            return {
                "error": "timeout",
                "had_validation_replan": False,
                "resolved_intent": resolved_intent_dict,
            }

        last_planner_response_id = getattr(response, "id", None)
        tcalls = output_items_to_tool_calls_compat(response)
        rtext = extract_response_text(response)
        if tcalls and rtext and preamble_callback:
            try:
                preamble_callback(rtext.strip())
            except Exception:
                pass
        msg = SimpleNamespace(
            content=rtext or None,
            tool_calls=tcalls,
        )

        if not tcalls:
            text = (rtext or "").strip()
            return {
                "error": None,
                "conversational_text": text,
                "validated_plan": None,
                "execution_results": None,
                "max_tokens_final": 0,
                "validation_failed": False,
                "validation_errors": [],
                "had_output_correction": False,
                "had_validation_replan": False,
                "planner_response_id": last_planner_response_id,
                "responses_synthesis": None,
                "resolved_intent": resolved_intent_dict,
            }

        raw_plan_calls = openai_tool_calls_to_plan_calls(tcalls)
        output_intent_fc = extract_output_intent_from_tool_args(tcalls)
        if not raw_plan_calls:
            text = (getattr(msg, "content", None) or rtext or "").strip()
            return {
                "error": None,
                "conversational_text": text or "No valid tool calls.",
                "validated_plan": None,
                "execution_results": None,
                "max_tokens_final": 0,
                "validation_failed": False,
                "validation_errors": [],
                "had_output_correction": False,
                "had_validation_replan": False,
                "planner_response_id": last_planner_response_id,
                "responses_synthesis": None,
                "resolved_intent": resolved_intent_dict,
            }

        plan: dict[str, Any] = {
            "tool_calls": raw_plan_calls,
            "output_intent": output_intent_fc,
            "reasoning": "openai_function_calling",
            "neighborhood_resolved": None,
            "combine_maps": False,
        }
        _infer_plan_neighborhood_resolved(plan)
        plan["combine_maps"] = _infer_combine_maps_from_tools(plan)

        validated = validate_plan(plan, live_schema_context)
        ve = validated.get("validation_errors") or []

        if plan.get("tool_calls") and not (validated.get("tool_calls") or []):
            if attempt < MAX_OUTPUT_CORRECTION_ATTEMPTS:
                correction_hint = format_validation_plan_correction(
                    ve, raw_plan_calls
                )
                had_validation_replan = True
                continue
            return {
                "error": None,
                "conversational_text": None,
                "validated_plan": validated,
                "execution_results": None,
                "max_tokens_final": 0,
                "validation_failed": True,
                "validation_errors": ve,
                "had_output_correction": False,
                "had_validation_replan": had_validation_replan,
                "planner_response_id": last_planner_response_id,
                "responses_synthesis": None,
                "resolved_intent": resolved_intent_dict,
            }

        execution_results = execute_plan(validated, engine, user_input)
        oi = normalize_output_intent(validated.get("output_intent") or "auto")
        issues = validate_output_completeness(user_input, oi, execution_results)

        if issues and attempt < MAX_OUTPUT_CORRECTION_ATTEMPTS:
            correction_hint = format_output_completeness_correction(issues, user_input, oi)
            had_output_correction = True
            continue

        for res in execution_results:
            res["renderer"] = decide_renderer(res, res.get("output_intent") or oi)

        last_tcalls = tcalls
        last_validated = validated
        last_execution = execution_results
        last_ve = ve
        completed_tool_path = True
        break

    # If loop never finished the tool path (e.g. exhausted replans), fail safe
    if not completed_tool_path:
        return {
            "error": None,
            "conversational_text": "Lo siento, no pude procesar tu mensaje. ¿Puedes reformular?",
            "validated_plan": None,
            "execution_results": None,
            "max_tokens_final": 0,
            "validation_failed": False,
            "validation_errors": [],
            "had_output_correction": had_output_correction,
            "had_validation_replan": had_validation_replan,
            "planner_response_id": None,
            "responses_synthesis": None,
            "resolved_intent": resolved_intent_dict,
        }

    execution_results = last_execution
    validated = last_validated
    ve = last_ve
    tcalls = last_tcalls

    confirmed = format_confirmed_visuals(execution_results)

    followup_user = (
        f"Escribe la respuesta final para el usuario con la voz de Rooster (máx. 3 frases, sin negritas markdown). "
        f"La interfaz mostrará después de tu texto estos visuales: {confirmed}. "
        f"El texto debe coincidir con ellos. Pregunta del usuario: {user_input!r}\n"
        f"Todo en **español** salvo que el usuario hubiera escrito solo en inglés.\n"
        f"Añade al final la línea <!-- FOLLOW_UPS: [...] --> exactamente como indica el system prompt del sintetizador."
    )
    max_final = infer_synth_max_tokens(execution_results) + 50

    pairs: list[tuple[str, str]] = []
    for tc in tcalls:
        er = next(
            (r for r in execution_results if r.get("tool_call_id") == tc.id),
            None,
        )
        if er is None:
            payload: dict[str, Any] = {
                "error": (
                    "Esta herramienta no se ejecutó (parámetros no válidos o barrio no reconocido)."
                )
            }
        else:
            summ = _build_results_summary_for_synth([er])
            payload = summ[0] if summ else {"error": "sin resultado"}
        pairs.append((str(tc.id), json.dumps(payload, default=str)))

    rsyn: dict[str, Any] | None = None
    if last_planner_response_id:
        rsyn = {
            "previous_response_id": last_planner_response_id,
            "input_items": tool_json_payloads_to_responses_input(pairs, followup_user),
            "instructions": SYNTHESISER_SYSTEM_PROMPT,
            "max_tokens": max_final,
        }

    return {
        "error": None,
        "conversational_text": None,
        "validated_plan": validated,
        "execution_results": execution_results,
        "max_tokens_final": max_final,
        "validation_failed": False,
        "validation_errors": ve,
        "had_output_correction": had_output_correction,
        "had_validation_replan": had_validation_replan,
        "planner_response_id": last_planner_response_id,
        "responses_synthesis": rsyn,
        "resolved_intent": resolved_intent_dict,
    }


def stream_openai_final_response_messages(
    messages: list[dict[str, Any]] | None,
    model: str,
    max_tokens: int,
    timeout_sec: float,
    *,
    responses_synthesis: dict[str, Any] | None = None,
) -> Iterator[str]:
    """Second-turn streaming completion (Responses API; chat path deprecated)."""
    if responses_synthesis:
        from agent.responses_api import (
            create_response_synthesis,
            get_openai_client,
            iter_response_stream_text,
        )

        model_name = _resolve_synthesiser_model(model)
        client = get_openai_client(max(timeout_sec, 120.0))
        stream = create_response_synthesis(
            client,
            model=model_name,
            instructions=str(responses_synthesis.get("instructions") or ""),
            input_items=list(responses_synthesis.get("input_items") or []),
            previous_response_id=str(responses_synthesis.get("previous_response_id") or ""),
            max_output_tokens=int(responses_synthesis.get("max_tokens") or max_tokens),
            stream=True,
        )
        yield from iter_response_stream_text(stream)
        return

    from agent.responses_api import (
        get_openai_client,
        iter_response_stream_text,
        reasoning_param_for_model,
        supports_temperature,
    )

    model_name = _resolve_synthesiser_model(model)
    client = get_openai_client(max(timeout_sec, 120.0))
    instructions = (messages or [{}])[0].get("content", SYNTHESISER_SYSTEM_PROMPT)
    user_text = "\n\n".join(
        str(m.get("content") or "") for m in (messages or [])[1:] if m.get("content")
    )
    kwargs: dict[str, Any] = {
        "model": model_name,
        "instructions": instructions,
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": user_text}],
            }
        ],
        "max_output_tokens": max_tokens,
        "stream": True,
    }
    if supports_temperature(model_name):
        kwargs["temperature"] = 0.3
    rpar = reasoning_param_for_model(model_name, "low")
    if rpar is not None:
        kwargs["reasoning"] = rpar
    stream = client.responses.create(**kwargs)
    yield from iter_response_stream_text(stream)


def run_synthesiser(
    question: str,
    plan: dict[str, Any],
    execution_results: list[dict[str, Any]],
    conversation_state: dict[str, Any],
    model: str | None,
    timeout_sec: float = SUMMARIZE_TIMEOUT_SEC,
    confirmed_visuals: str | None = None,
    max_tokens_override: int | None = None,
) -> str:
    """Stage 4: narrative from LLM."""
    messages, max_tok = build_synthesiser_messages(
        question,
        plan,
        execution_results,
        conversation_state,
        confirmed_visuals=confirmed_visuals,
        max_tokens_override=max_tokens_override,
    )
    model_name = _resolve_synthesiser_model(model)

    def _run() -> str:
        return _llm_synthesiser(messages, model_name, timeout_sec, max_tokens=max_tok)

    try:
        with ThreadPoolExecutor(max_workers=1) as ex:
            return ex.submit(_run).result(timeout=timeout_sec)
    except FuturesTimeoutError:
        from agent import ui_es as UI

        return UI.SYNTH_TIMEOUT_BODY


def update_conversation_state(
    state: dict[str, Any],
    question: str,
    plan: dict[str, Any],
    execution_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Update mutable conversation state after a turn."""
    if plan.get("neighborhood_resolved"):
        nb = str(plan["neighborhood_resolved"]).strip()
        if nb and nb not in state["neighborhoods_discussed"]:
            state["neighborhoods_discussed"].append(nb)
    for call in plan.get("tool_calls") or []:
        if not isinstance(call, dict):
            continue
        p = call.get("params") or {}
        op = p.get("operation")
        if op and op != "both":
            state["operation_focus"] = op
        mp = p.get("max_price")
        if mp is not None:
            try:
                state["price_ceiling"] = int(mp)
            except (TypeError, ValueError):
                pass
    state["turns"] = int(state.get("turns", 0)) + 1
    if state["turns"] <= 2:
        state["stage"] = "orienting"
    elif len(state.get("neighborhoods_discussed", [])) >= 2:
        state["stage"] = "evaluating"
    elif state["turns"] >= 6:
        state["stage"] = "deciding"
    if plan.get("tool_calls"):
        state["last_intent"] = plan["tool_calls"][0].get("tool")
    priority_keywords = {
        "yield": ["yield", "rentabilidad", "rendimiento", "retorno"],
        "transport": ["transport", "metro", "bus", "conectividad"],
        "price": ["precio", "barato", "económico", "budget", "coste"],
        "tourism": ["turístico", "airbnb", "vut", "corta estancia"],
    }
    q_lower = (question or "").lower()
    for priority, kws in priority_keywords.items():
        if any(kw in q_lower for kw in kws):
            if priority not in state["user_priorities"]:
                state["user_priorities"].append(priority)
    return state


def map_renderer_to_dispatch_intent(renderer: str) -> str:
    """Map abstract renderer name to existing RENDERERS keys in renderers.py."""
    mapping = {
        "table": "search",
        "point_map": "geo",
        "transit_map": "transit_map",
        "tourism_map": "tourism_map",
        "combined_map": "combined_map",
        "bar_chart": "ranking",
        "profile_scatter": "profile_scatter",
        "metric_cards": "overview",
        "chart": "chart",
        "neighborhood_highlight_map": "neighborhood_highlight",
        "no_coords_fallback": "no_coords",
        "empty": "search",
    }
    return mapping.get(renderer, "search")


def build_render_stack(
    validated_plan: dict[str, Any],
    execution_results: list[dict[str, Any]],
    geo_key: int,
) -> list[dict[str, Any]]:
    """
    Build render blocks for dispatch. When combine_maps is True, merge spatial tools
    into one combined_map block, then append non-spatial tools (e.g. charts, profiles).
    """
    combine = bool(validated_plan.get("combine_maps"))
    spatial_tools = {"query_listings", "query_transit_stops", "query_tourist_apartments"}
    blocks: list[dict[str, Any]] = []
    if not execution_results:
        return blocks

    spatial_merged = False
    if combine:
        rl: list = []
        rt: list = []
        ru: list = []
        for res in execution_results:
            if not res.get("success"):
                continue
            t = res.get("tool")
            r = res.get("rows") or []
            if t == "query_listings":
                rl = r
            elif t == "query_transit_stops":
                rt = r
            elif t == "query_tourist_apartments":
                ru = r
        if rl or rt or ru:
            blocks.append(
                {
                    "intent": "combined_map",
                    "rows": [],
                    "meta": {
                        "geo_key": geo_key,
                        "rows_listings": rl,
                        "rows_transit": rt,
                        "rows_tourism": ru,
                        "caveat": "",
                        "tool": "__combined_map__",
                        "dedup_params": "",
                    },
                }
            )
            spatial_merged = True

    for res in execution_results:
        if combine and spatial_merged and res.get("tool") in spatial_tools:
            continue
        ren = (res.get("renderer") or "").strip()
        if ren == "empty":
            continue
        intent = map_renderer_to_dispatch_intent(ren)
        rows = res.get("rows") or []
        p = res.get("params") or {}
        meta: dict[str, Any] = {
            "geo_key": geo_key,
            "caveat": "",
            "tool": (res.get("tool") or "") or "",
            "dedup_params": _dedup_params_signature(p if isinstance(p, dict) else {}),
        }
        if intent == "chart":
            ct = (p.get("chart_type") or "scatter").strip().lower()
            if ct not in ("scatter", "amenity", "floor"):
                ct = "scatter"
            meta["chart_type"] = ct
        if intent == "ranking":
            meta["metric_label"] = "Value"
        if intent == "ranking" and rows and "value" not in rows[0]:
            for r in rows:
                if "value" not in r and r.get("investment_score") is not None:
                    r["value"] = r["investment_score"]
        blocks.append({"intent": intent, "rows": rows, "meta": meta})
    return _dedupe_render_blocks(blocks)
