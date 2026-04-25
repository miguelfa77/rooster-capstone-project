"""
Bounded agent loop configuration (v2). Multi-step tool rounds may extend this module.

The production path currently uses ``run_openai_function_calling_pipeline`` in
``agent_pipeline``; ``ROOSTER_MAX_AGENT_STEPS`` is reserved for future multi-round expansion.
"""

from __future__ import annotations

import os
from typing import Any

MAX_STEPS_DEFAULT = int(os.environ.get("ROOSTER_MAX_AGENT_STEPS", "3"))


def compute_max_steps(resolved_intent: dict[str, Any] | None) -> int:
    if not resolved_intent:
        return min(MAX_STEPS_DEFAULT, 3)
    if resolved_intent.get("comparison_mode") == "find_best" or resolved_intent.get(
        "recommendation_requested"
    ):
        return min(5, max(MAX_STEPS_DEFAULT, 5))
    return min(MAX_STEPS_DEFAULT, 3)


def run_agent_loop(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """
    Backwards-compatible entry: delegates to the single-pass FC + synthesis pipeline
    until multi-round Responses state is fully wired.
    """
    from agent.agent_pipeline import run_openai_function_calling_pipeline

    return run_openai_function_calling_pipeline(*args, **kwargs)
