"""Rooster agent API: conversations and streamed turns over Server-Sent Events.

Called only by trusted servers (the web app's server routes and the voice service),
never directly by a browser, so every route except /health requires X-Rooster-Key.
User routes also require X-Rooster-User (a user id from PUT /users); a conversation
owned by another user is reported as not found.
"""

from __future__ import annotations

import hmac
import logging
import os
import threading
import time
import uuid
from contextlib import ExitStack
from functools import lru_cache
from typing import Any, Callable, Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field, field_validator

from agent import intel_data
from agent import ui_es as UI
from agent.agent_pipeline import get_live_schema_context
from agent.config import SYNTHESIZER_MODEL_DEFAULT
from agent.llm_sql import get_pg_engine, get_schema_context
from agent.turn import DoneEvent, ErrorEvent, TurnContext, run_turn
from api.models import (
    ConversationOut,
    ConversationRename,
    ConversationSummary,
    ListingRow,
    MarketFraming,
    NeighborhoodsOut,
    Preferences,
    RankingRow,
    TourismRow,
    TransitRow,
    UserIn,
    UserOut,
    YieldLiquidityRow,
)
from api.sse import stream_turn, to_json
from api.store import ConversationBusy, ConversationStore

log = logging.getLogger(__name__)

TURN_TIMEOUT_SEC = 60.0
DEFAULT_MAX_TURNS_PER_MINUTE = 30
LIVE_SCHEMA_TTL_SEC = 300

app = FastAPI(title="Rooster API", version="3.0.0")


# ── Dependencies ──────────────────────────────────────────────────────────


def require_key(x_rooster_key: str | None = Header(default=None)) -> None:
    expected = os.getenv("ROOSTER_API_KEY") or ""
    if not expected or not hmac.compare_digest((x_rooster_key or "").encode(), expected.encode()):
        raise HTTPException(status_code=401, detail="unauthorized")


@lru_cache(maxsize=1)
def get_store() -> ConversationStore:
    return ConversationStore(get_pg_engine())


def require_user(
    _: None = Depends(require_key),
    x_rooster_user: str | None = Header(default=None),
    store: ConversationStore = Depends(get_store),
) -> uuid.UUID:
    """The calling user's id from X-Rooster-User; 401 when missing, malformed or unknown."""
    try:
        user_id = uuid.UUID(x_rooster_user or "")
    except ValueError:
        raise HTTPException(status_code=401, detail="user_required") from None
    if store.get_user(user_id) is None:
        raise HTTPException(status_code=401, detail="user_required")
    return user_id


@lru_cache(maxsize=1)
def _static_schema() -> str:
    return get_schema_context()


_live_schema_cache: dict = {"value": None, "at": 0.0}
_live_schema_lock = threading.Lock()


def _live_schema() -> str:
    with _live_schema_lock:
        if _live_schema_cache["value"] is None or time.monotonic() - _live_schema_cache["at"] > LIVE_SCHEMA_TTL_SEC:
            _live_schema_cache["value"] = get_live_schema_context()
            _live_schema_cache["at"] = time.monotonic()
        return _live_schema_cache["value"]


def get_turn_context() -> TurnContext:
    return TurnContext(
        engine=get_pg_engine(),
        live_schema=_live_schema(),
        static_schema=_static_schema(),
        model=SYNTHESIZER_MODEL_DEFAULT,
        timeout_sec=TURN_TIMEOUT_SEC,
    )


_warned_bad_turn_limit = False


def _max_turns_per_minute() -> int:
    """ROOSTER_MAX_TURNS_PER_MINUTE as an int; unset or invalid → 30 (invalid warns once), negative → 0 (off)."""
    global _warned_bad_turn_limit
    raw = os.getenv("ROOSTER_MAX_TURNS_PER_MINUTE")
    if not raw:
        return DEFAULT_MAX_TURNS_PER_MINUTE
    try:
        return max(int(raw), 0)
    except ValueError:
        if not _warned_bad_turn_limit:
            _warned_bad_turn_limit = True
            log.warning("Invalid ROOSTER_MAX_TURNS_PER_MINUTE=%r; using %d", raw, DEFAULT_MAX_TURNS_PER_MINUTE)
        return DEFAULT_MAX_TURNS_PER_MINUTE


def _parse_id(conversation_id: str) -> uuid.UUID:
    try:
        return uuid.UUID(conversation_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="not found") from None


# ── Routes ────────────────────────────────────────────────────────────────


class MessageIn(BaseModel):
    text: str = Field(max_length=2000)

    @field_validator("text")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text must not be blank")
        return v.strip()


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.put("/users", response_model=UserOut, response_model_exclude_unset=True, dependencies=[Depends(require_key)])
def put_user(body: UserIn, store: ConversationStore = Depends(get_store)) -> dict:
    return store.upsert_user(body.email, body.name, body.image)


@app.get("/me", response_model=UserOut, response_model_exclude_unset=True)
def get_me(user_id: uuid.UUID = Depends(require_user), store: ConversationStore = Depends(get_store)) -> dict:
    return store.get_user(user_id)


@app.patch("/me/preferences", response_model=Preferences, response_model_exclude_unset=True)
def patch_preferences(
    body: Preferences, user_id: uuid.UUID = Depends(require_user), store: ConversationStore = Depends(get_store)
) -> dict:
    return store.update_preferences(user_id, body.model_dump(exclude_none=True))


@app.get("/conversations", response_model=list[ConversationSummary])
def list_conversations(
    user_id: uuid.UUID = Depends(require_user), store: ConversationStore = Depends(get_store)
) -> list[dict]:
    return store.list_conversations(user_id)


@app.post("/conversations", status_code=201)
def create_conversation(
    user_id: uuid.UUID = Depends(require_user), store: ConversationStore = Depends(get_store)
) -> dict:
    return {"id": str(store.create(user_id))}


@app.get("/conversations/{conversation_id}", response_model=ConversationOut)
def get_conversation(
    conversation_id: str, user_id: uuid.UUID = Depends(require_user), store: ConversationStore = Depends(get_store)
) -> dict:
    cid = _parse_id(conversation_id)
    summary, state = store.summary(cid, user_id), store.load(cid, user_id)
    if summary is None or state is None:
        raise HTTPException(status_code=404, detail="not found")
    return {"id": str(cid), "title": summary["title"], "messages": state.messages}


@app.patch("/conversations/{conversation_id}", response_model=ConversationSummary)
def rename_conversation(
    conversation_id: str,
    body: ConversationRename,
    user_id: uuid.UUID = Depends(require_user),
    store: ConversationStore = Depends(get_store),
) -> dict:
    cid = _parse_id(conversation_id)
    if not store.rename(cid, user_id, body.title):
        raise HTTPException(status_code=404, detail="not found")
    return store.summary(cid, user_id)


@app.delete("/conversations/{conversation_id}", status_code=204)
def delete_conversation(
    conversation_id: str, user_id: uuid.UUID = Depends(require_user), store: ConversationStore = Depends(get_store)
) -> Response:
    if not store.delete(_parse_id(conversation_id), user_id):
        raise HTTPException(status_code=404, detail="not found")
    return Response(status_code=204)


@app.post("/conversations/{conversation_id}/messages")
def post_message(
    conversation_id: str,
    body: MessageIn,
    user_id: uuid.UUID = Depends(require_user),
    store: ConversationStore = Depends(get_store),
    ctx: TurnContext = Depends(get_turn_context),
) -> StreamingResponse:
    cid = _parse_id(conversation_id)
    if store.load(cid, user_id) is None:
        raise HTTPException(status_code=404, detail="not found")
    cap = _max_turns_per_minute()
    if cap and not store.try_start_turn(cap):
        raise HTTPException(status_code=429, detail="rate_limited")

    held = ExitStack()
    try:
        held.enter_context(store.lock(cid))
    except ConversationBusy:
        raise HTTPException(status_code=409, detail="busy") from None

    def run(emit) -> None:
        """Run the turn, save it, then send `done` so a client never sees done before the save."""
        try:
            def forward(event) -> None:
                if not isinstance(event, DoneEvent):
                    emit(event)

            new_state = run_turn(body.text, state, ctx, forward)
            store.save(cid, new_state)
        except Exception as e:
            emit(ErrorEvent(reason=str(e), suggestion=UI.CHAT_EXCEPTION_SUGGESTION))
        finally:
            held.close()
        emit(DoneEvent())

    try:  # anything failing before the stream starts must not leave the lock held
        state = store.load(cid, user_id)
        return StreamingResponse(
            stream_turn(run),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    except BaseException:
        held.close()
        raise


# ── Intel (read-only data for the dashboard) ──────────────────────────────

_intel_cache: dict[tuple, tuple[float, str]] = {}
_intel_cache_lock = threading.Lock()


def clear_intel_cache() -> None:
    with _intel_cache_lock:
        _intel_cache.clear()


def _cached_json(key: tuple, ttl_sec: int, load: Callable[[], Any]) -> Response:
    """Serve a JSON response, reusing it for ttl_sec (same TTLs as the Streamlit app)."""
    with _intel_cache_lock:
        hit = _intel_cache.get(key)
        if hit and time.monotonic() - hit[0] < ttl_sec:
            return Response(content=hit[1], media_type="application/json")
    data = load()
    if hasattr(data, "to_dict"):  # DataFrame → list of row dicts
        data = data.to_dict(orient="records")
    body = to_json(data)
    with _intel_cache_lock:
        _intel_cache[key] = (time.monotonic(), body)
    return Response(content=body, media_type="application/json")


_VALID_METRICS = set(UI.CHOROPLETH_METRICS.values())


def _doc(model: Any) -> dict:
    """Document a 200 body; intel routes still return the cached Response unchanged."""
    return {200: {"model": model}}


@app.get("/intel/framing", responses=_doc(MarketFraming | None), dependencies=[Depends(require_key)])
def intel_framing() -> Response:
    return _cached_json(("framing",), 300, lambda: intel_data.market_framing())


@app.get("/intel/neighborhoods", responses=_doc(NeighborhoodsOut), dependencies=[Depends(require_key)])
def intel_neighborhoods(metric: str = Query(...)) -> Response:
    if metric not in _VALID_METRICS:
        raise HTTPException(status_code=422, detail=f"metric must be one of {sorted(_VALID_METRICS)}")

    def load() -> dict:
        geojson, vmin, vmax = intel_data.neighborhood_geojson(metric)
        return {"geojson": geojson, "vmin": vmin, "vmax": vmax}

    return _cached_json(("neighborhoods", metric), 300, load)


@app.get("/intel/overlays/transit", responses=_doc(list[TransitRow]), dependencies=[Depends(require_key)])
def intel_transit() -> Response:
    return _cached_json(("transit",), 600, lambda: intel_data.transit_overlay())


@app.get("/intel/overlays/tourism", responses=_doc(list[TourismRow]), dependencies=[Depends(require_key)])
def intel_tourism() -> Response:
    return _cached_json(("tourism",), 600, lambda: intel_data.tourism_overlay())


@app.get("/intel/listings", responses=_doc(list[ListingRow]), dependencies=[Depends(require_key)])
def intel_listings(operation: Literal["All", "venta", "alquiler"] = "All") -> Response:
    return _cached_json(("listings", operation), 300, lambda: intel_data.listings(operation))


@app.get("/intel/parcels", responses=_doc(dict[str, Any]), dependencies=[Depends(require_key)])
def intel_parcels() -> Response:
    return _cached_json(("parcels",), 600, lambda: intel_data.parcels_geojson())


@app.get("/intel/ranking", responses=_doc(list[RankingRow]), dependencies=[Depends(require_key)])
def intel_ranking() -> Response:
    return _cached_json(("ranking",), 300, lambda: intel_data.neighborhood_ranking())


@app.get("/intel/yield-liquidity", responses=_doc(list[YieldLiquidityRow]), dependencies=[Depends(require_key)])
def intel_yield_liquidity() -> Response:
    return _cached_json(("yield_liquidity",), 300, lambda: intel_data.yield_liquidity())
