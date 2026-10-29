import asyncio
import json
import threading
import uuid
from datetime import datetime
from decimal import Decimal

import numpy as np
from shapely.geometry import Point

from agent.turn import DoneEvent, TextDeltaEvent, ThoughtEvent
from api.sse import format_event, stream_turn, to_json


def test_to_json_handles_database_types():
    out = json.loads(to_json({"d": Decimal("6.2"), "t": datetime(2026, 1, 1), "g": Point(0, 1)}))
    assert out == {"d": 6.2, "t": "2026-01-01T00:00:00", "g": {"type": "Point", "coordinates": [0.0, 1.0]}}


def test_to_json_handles_numpy_uuid_and_nan():
    cid = uuid.uuid4()
    out = json.loads(to_json({"n": np.int64(3), "f": np.float64(1.5), "u": cid, "nan": float("nan"), "l": [np.nan]}))
    assert out == {"n": 3, "f": 1.5, "u": str(cid), "nan": None, "l": [None]}


def test_to_json_is_compact_and_keeps_accents():
    assert to_json({"a": "ñ", "b": [1, 2]}) == '{"a":"ñ","b":[1,2]}'


def test_format_event_is_sse_frame():
    assert format_event(ThoughtEvent(text="x")) == 'event: thought\ndata: {"type":"thought","text":"x"}\n\n'


def _collect(agen, limit=None):
    async def go():
        frames = []
        async for frame in agen:
            frames.append(frame)
            if limit and len(frames) >= limit:
                break
        await agen.aclose()
        return frames

    return asyncio.run(go())


def test_stream_turn_yields_until_done():
    def run(emit):
        emit(ThoughtEvent(text="a"))
        emit(TextDeltaEvent(text="b"))
        emit(DoneEvent())

    frames = _collect(stream_turn(run))
    assert [f.split("\n")[0] for f in frames] == ["event: thought", "event: text_delta", "event: done"]


def test_stream_turn_reports_unexpected_exception_then_done():
    def run(emit):
        raise RuntimeError("kaput")

    frames = _collect(stream_turn(run))
    assert frames[0].startswith("event: error") and "kaput" in frames[0]
    assert frames[-1].startswith("event: done")


def test_stream_turn_completes_after_disconnect():
    finished = threading.Event()
    release = threading.Event()

    def run(emit):
        emit(ThoughtEvent(text="first"))
        release.wait(2)
        emit(TextDeltaEvent(text="late"))
        emit(DoneEvent())
        finished.set()

    frames = _collect(stream_turn(run), limit=1)
    release.set()
    assert frames[0].startswith("event: thought")
    assert finished.wait(2)


def test_to_json_maps_pandas_nulls_to_null():
    import pandas as pd

    rows = pd.DataFrame({"d": pd.to_datetime(["2026-01-01", None])}).to_dict("records")
    assert json.loads(to_json(rows)) == [{"d": "2026-01-01T00:00:00"}, {"d": None}]
    assert json.loads(to_json({"na": pd.NA})) == {"na": None}


def test_to_json_handles_numpy_arrays():
    assert json.loads(to_json({"a": np.array([1, 2])})) == {"a": [1, 2]}
