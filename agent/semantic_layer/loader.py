"""Load the hand-authored semantic registry."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from agent.semantic_layer.models import SemanticRegistry


@lru_cache(maxsize=4)
def load_registry(path: str | None = None) -> SemanticRegistry:
    registry_path = Path(path) if path else Path(__file__).with_name("registry.json")
    data = json.loads(registry_path.read_text(encoding="utf-8"))
    return SemanticRegistry.model_validate(data)
