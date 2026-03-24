# utils/run_session.py
"""Per-run scrape progress: resume only after an interrupted run, not from CSV max page."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from .. import config


def _path() -> Path:
    return Path(config.CHECKPOINT_FILE)


def _default_op() -> Dict[str, Any]:
    return {"in_progress": False, "last_completed_page": 0}


def load_all() -> Dict[str, Any]:
    p = _path()
    if not p.exists():
        return {}
    try:
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def save_all(data: Dict[str, Any]) -> None:
    p = _path()
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def clear_all() -> None:
    p = _path()
    if p.exists():
        try:
            p.unlink()
        except OSError:
            pass


def get_operation(operation: str) -> Dict[str, Any]:
    raw = load_all().get(operation)
    if not isinstance(raw, dict):
        return _default_op()
    out = _default_op()
    out["in_progress"] = bool(raw.get("in_progress", False))
    try:
        out["last_completed_page"] = max(0, int(raw.get("last_completed_page", 0)))
    except (TypeError, ValueError):
        out["last_completed_page"] = 0
    return out


def set_page_completed(operation: str, page: int) -> None:
    data = load_all()
    cur = data.get(operation) if isinstance(data.get(operation), dict) else {}
    merged = _default_op()
    merged.update(cur)
    merged["in_progress"] = True
    merged["last_completed_page"] = max(0, int(page))
    data[operation] = merged
    save_all(data)


def mark_operation_complete(operation: str) -> None:
    data = load_all()
    cur = data.get(operation) if isinstance(data.get(operation), dict) else {}
    merged = _default_op()
    merged.update(cur)
    merged["in_progress"] = False
    data[operation] = merged
    save_all(data)
