"""HTTP client for the sandbox service that runs `code` primitives. No UI dependencies."""

from __future__ import annotations

import os
from typing import Any

import requests


def _normalize_sandbox_url(url: str) -> str:
    url = (url or "").strip()
    if url and not url.startswith(("http://", "https://")):
        url = "http://" + url
    return url or "http://localhost:8001"


SANDBOX_URL = _normalize_sandbox_url(os.getenv("ROOSTER_SANDBOX_URL", "http://localhost:8001"))


def call_sandbox(code: str, data: list[dict[str, Any]]) -> dict[str, Any]:
    """Execute code in the sandbox service. Pure HTTP — no Streamlit."""
    try:
        response = requests.post(
            f"{SANDBOX_URL}/execute",
            json={"code": code, "data": data},
            timeout=35,
        )
        response.raise_for_status()
        return response.json()
    except Exception as exc:
        return {"success": False, "output_type": None, "output": None, "error": str(exc)}
