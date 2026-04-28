"""Structured per-stage logging helpers."""

from __future__ import annotations

import json
import logging
from typing import Any

_LOG = logging.getLogger("rooster.stage")


def _safe_payload(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, default=str, sort_keys=True)


def log_stage(stage: str, event: str, **payload: Any) -> None:
    """Emit one machine-readable log line for a pipeline stage."""
    _LOG.info(
        "rooster_stage=%s event=%s payload=%s",
        stage,
        event,
        _safe_payload(payload),
    )
