"""Deterministic semantic layer for Rooster v2."""

from agent.semantic_layer.models import ResolvedQuery
from agent.semantic_layer.resolver import resolve_query

__all__ = ["ResolvedQuery", "resolve_query"]
