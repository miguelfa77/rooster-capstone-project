from __future__ import annotations

import io
import sys
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Any

import branca.colormap as cm
import folium
import geopandas as gpd
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field
from shapely.geometry import shape

EXEC_TIMEOUT_SEC = 30

# Injected into every execution — packages already imported at module level,
# so workers inherited via fork (Linux/Railway) pay zero import cost.
_BASE_NAMESPACE: dict[str, Any] = {
    "pd": pd,
    "folium": folium,
    "gpd": gpd,
    "px": px,
    "go": go,
    "pio": pio,
    "cm": cm,
    "shape": shape,
}

_POOL: ProcessPoolExecutor | None = None


class ExecuteRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    code: str
    data: list[dict[str, Any]] = Field(default_factory=list)


class ExecuteResponse(BaseModel):
    success: bool
    output_type: str | None = None
    output: str | None = None
    error: str | None = None


def _run_code(code: str, data: list[dict[str, Any]]) -> dict[str, Any]:
    """Runs in a pooled worker. All heavy packages already in _BASE_NAMESPACE."""
    capture = io.StringIO()
    sys.stdout = capture
    try:
        namespace = {**_BASE_NAMESPACE, "df": pd.DataFrame(data)}
        exec(code, namespace)  # noqa: S102
        raw = capture.getvalue().strip()
        output_type = None
        if raw.startswith("{") and '"data"' in raw and '"layout"' in raw:
            output_type = "plotly_json"
        elif "<html" in raw.lower() and "folium" in raw.lower():
            output_type = "folium_html"
        return {"success": True, "output_type": output_type, "output": raw, "error": None}
    except Exception as exc:
        return {"success": False, "output_type": None, "output": None, "error": str(exc)}
    finally:
        sys.stdout = sys.__stdout__


def _get_pool() -> ProcessPoolExecutor:
    global _POOL
    if _POOL is None:
        _POOL = ProcessPoolExecutor(max_workers=2)
    return _POOL


def execute(code: str, data: list[dict[str, Any]]) -> dict[str, Any]:
    future = _get_pool().submit(_run_code, code, data)
    try:
        return future.result(timeout=EXEC_TIMEOUT_SEC)
    except FuturesTimeoutError:
        future.cancel()
        return {
            "success": False,
            "output_type": None,
            "output": None,
            "error": f"Execution timed out after {EXEC_TIMEOUT_SEC} seconds",
        }
    except Exception as exc:
        return {"success": False, "output_type": None, "output": None, "error": str(exc)}


app = FastAPI(title="Rooster Sandbox Service")


@app.on_event("startup")
def _warmup() -> None:
    """Spin up worker processes and pre-import packages before the first real request."""
    pool = _get_pool()
    fut = pool.submit(_run_code, "pass", [])
    try:
        fut.result(timeout=30)
    except Exception:
        pass


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/execute", response_model=ExecuteResponse)
def execute_route(payload: ExecuteRequest) -> ExecuteResponse:
    result = execute(payload.code, payload.data)
    return ExecuteResponse(**result)
