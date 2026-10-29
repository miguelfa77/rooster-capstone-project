"""JSON encoding and the worker-thread → Server-Sent Events bridge."""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from typing import Any, AsyncIterator, Callable

from pydantic import BaseModel

from agent import ui_es as UI
from agent.jsonable import to_jsonable
from agent.turn import DoneEvent, ErrorEvent

_LOG = logging.getLogger("rooster.api.sse")


def to_json(obj: Any) -> str:
    return json.dumps(to_jsonable(obj), separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def format_event(event: BaseModel) -> str:
    return f"event: {event.type}\ndata: {to_json(event)}\n\n"


_SENTINEL = object()


async def stream_turn(run: Callable[[Callable[[Any], None]], None]) -> AsyncIterator[str]:
    """Run `run(emit)` in a worker thread and yield its events as SSE frames until `done`.

    The worker is not tied to the client: if the client disconnects, it still runs to
    completion so the turn gets saved.
    """
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def emit(event: Any) -> None:
        try:
            loop.call_soon_threadsafe(queue.put_nowait, event)
        except RuntimeError:  # client gone and its event loop closed; keep running the turn
            pass

    def worker() -> None:
        try:
            run(emit)
        except Exception as e:
            _LOG.exception("Turn worker failed")
            emit(ErrorEvent(reason=str(e), suggestion=UI.CHAT_EXCEPTION_SUGGESTION))
            emit(DoneEvent())
        finally:
            try:
                loop.call_soon_threadsafe(queue.put_nowait, _SENTINEL)
            except RuntimeError:  # event loop already closed
                pass

    threading.Thread(target=worker, name="rooster-turn", daemon=False).start()
    while True:
        event = await queue.get()
        if event is _SENTINEL:
            return
        yield format_event(event)
        if event.type == "done":
            return
