from __future__ import annotations

import io
import multiprocessing as mp
import sys
from typing import Any

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

EXEC_TIMEOUT_SEC = 15


class ExecuteRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    code: str
    data: list[dict[str, Any]] = Field(default_factory=list)


class ExecuteResponse(BaseModel):
    success: bool
    output_type: str | None = None
    output: str | None = None
    error: str | None = None


def _worker(code: str, data: list[dict[str, Any]], queue: mp.Queue) -> None:
    df = pd.DataFrame(data)
    namespace: dict[str, Any] = {"df": df, "pd": pd}
    capture = io.StringIO()
    previous_stdout = sys.stdout
    sys.stdout = capture
    try:
        exec(code, namespace)
        raw_output = capture.getvalue().strip()
        output_type = None
        if raw_output.startswith("{") and '"data"' in raw_output and '"layout"' in raw_output:
            output_type = "plotly_json"
        elif "<html" in raw_output.lower() and "folium" in raw_output.lower():
            output_type = "folium_html"
        queue.put(
            {
                "success": True,
                "output_type": output_type,
                "output": raw_output,
                "error": None,
            }
        )
    except Exception as exc:
        queue.put(
            {
                "success": False,
                "output_type": None,
                "output": None,
                "error": str(exc),
            }
        )
    finally:
        sys.stdout = previous_stdout


def execute(code: str, data: list[dict[str, Any]]) -> dict[str, Any]:
    queue: mp.Queue = mp.Queue(maxsize=1)
    process = mp.Process(target=_worker, args=(code, data, queue), daemon=True)
    process.start()
    process.join(EXEC_TIMEOUT_SEC)
    if process.is_alive():
        process.terminate()
        process.join(timeout=1)
        return {
            "success": False,
            "output_type": None,
            "output": None,
            "error": f"Execution timed out after {EXEC_TIMEOUT_SEC} seconds",
        }
    if queue.empty():
        return {
            "success": False,
            "output_type": None,
            "output": None,
            "error": "Execution failed without output",
        }
    return queue.get()


app = FastAPI(title="Rooster Sandbox Service")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/execute", response_model=ExecuteResponse)
def execute_route(payload: ExecuteRequest) -> ExecuteResponse:
    result = execute(payload.code, payload.data)
    return ExecuteResponse(**result)
