"""Rooster agent API: conversations and streamed turns over Server-Sent Events.

Called only by trusted servers (the web app's server routes and the voice service),
never directly by a browser, so every route except /health requires X-Rooster-Key.
"""

from __future__ import annotations

import hmac
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
from api.sse import stream_turn, to_json
from api.store import ConversationBusy, ConversationStore

TURN_TIMEOUT_SEC = 60.0
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


@app.post("/conversations", status_code=201, dependencies=[Depends(require_key)])
def create_conversation(store: ConversationStore = Depends(get_store)) -> dict:
    return {"id": str(store.create())}


@app.get("/conversations/{conversation_id}", dependencies=[Depends(require_key)])
def get_conversation(conversation_id: str, store: ConversationStore = Depends(get_store)) -> dict:
    cid = _parse_id(conversation_id)
    state = store.load(cid)
    if state is None:
        raise HTTPException(status_code=404, detail="not found")
    return {"id": str(cid), "messages": state.messages}


@app.post("/conversations/{conversation_id}/messages", dependencies=[Depends(require_key)])
def post_message(
    conversation_id: str,
    body: MessageIn,
    store: ConversationStore = Depends(get_store),
    ctx: TurnContext = Depends(get_turn_context),
) -> StreamingResponse:
    cid = _parse_id(conversation_id)
    if store.load(cid) is None:
        raise HTTPException(status_code=404, detail="not found")

    held = ExitStack()
    try:
        held.enter_context(store.lock(cid))
    except ConversationBusy:
        raise HTTPException(status_code=409, detail="busy") from None
    state = store.load(cid)

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

    return StreamingResponse(
        stream_turn(run),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


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


@app.get("/intel/framing", dependencies=[Depends(require_key)])
def intel_framing() -> Response:
    return _cached_json(("framing",), 300, lambda: intel_data.market_framing())


@app.get("/intel/neighborhoods", dependencies=[Depends(require_key)])
def intel_neighborhoods(metric: str = Query(...)) -> Response:
    if metric not in _VALID_METRICS:
        raise HTTPException(status_code=422, detail=f"metric must be one of {sorted(_VALID_METRICS)}")

    def load() -> dict:
        geojson, vmin, vmax = intel_data.neighborhood_geojson(metric)
        return {"geojson": geojson, "vmin": vmin, "vmax": vmax}

    return _cached_json(("neighborhoods", metric), 300, load)


@app.get("/intel/overlays/transit", dependencies=[Depends(require_key)])
def intel_transit() -> Response:
    return _cached_json(("transit",), 600, lambda: intel_data.transit_overlay())


@app.get("/intel/overlays/tourism", dependencies=[Depends(require_key)])
def intel_tourism() -> Response:
    return _cached_json(("tourism",), 600, lambda: intel_data.tourism_overlay())


@app.get("/intel/listings", dependencies=[Depends(require_key)])
def intel_listings(operation: Literal["All", "venta", "alquiler"] = "All") -> Response:
    return _cached_json(("listings", operation), 300, lambda: intel_data.listings(operation))


@app.get("/intel/parcels", dependencies=[Depends(require_key)])
def intel_parcels() -> Response:
    return _cached_json(("parcels",), 600, lambda: intel_data.parcels_geojson())


@app.get("/intel/ranking", dependencies=[Depends(require_key)])
def intel_ranking() -> Response:
    return _cached_json(("ranking",), 300, lambda: intel_data.neighborhood_ranking())


@app.get("/intel/yield-liquidity", dependencies=[Depends(require_key)])
def intel_yield_liquidity() -> Response:
    return _cached_json(("yield_liquidity",), 300, lambda: intel_data.yield_liquidity())
