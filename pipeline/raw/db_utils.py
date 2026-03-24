"""
Common PostgreSQL connection utilities for Rooster raw loaders.

Reads connection settings in this order:
  - ``agent/.env`` (if present) — sets every key defined in the file
  - ``pipeline/.env`` — merges unset keys (and overwrites empty ``PG*`` / ``DATABASE_URL``)
  - repo root ``.env`` — same merge rules as ``pipeline/.env``
  - environment variables already exported in the shell

Then connects using ``DATABASE_URL`` if set, otherwise
``PGHOST``, ``PGPORT``, ``PGUSER``, ``PGPASSWORD``, ``PGDATABASE``.

Also loads **repo root** ``.env`` (same merge rules as ``pipeline/.env``), so a single
``.env`` at project root works for CLI loaders — not only ``agent/.env`` / ``pipeline/.env``.
"""

import os
from pathlib import Path
from typing import Any

import psycopg2

_PG_ENV_KEYS = frozenset(
    {"DATABASE_URL", "PGHOST", "PGPORT", "PGUSER", "PGPASSWORD", "PGDATABASE"}
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _should_merge_dotenv_key(key: str) -> bool:
    """True if we may set *key* from a merge-only .env (pipeline or repo root)."""
    if key not in os.environ:
        return True
    if key in _PG_ENV_KEYS and os.environ.get(key) == "":
        return True
    return False


def _load_agent_dotenv_override() -> None:
    """Same semantics as ``agent/llm_sql._load_agent_env``: agent/.env wins over prior env."""
    dotenv = _repo_root() / "agent" / ".env"
    if not dotenv.exists():
        return
    for line in dotenv.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        if key:
            os.environ[key] = value


def _load_env_from_pipeline_dotenv() -> None:
    """
    Minimal loader for pipeline/.env that contains shell-style `export KEY=VALUE` lines.
    """
    pipeline_root = Path(__file__).resolve().parents[1]
    dotenv_path = pipeline_root / ".env"
    if not dotenv_path.exists():
        return
    for line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        if key and _should_merge_dotenv_key(key):
            os.environ[key] = value


def _load_env_from_repo_root_dotenv() -> None:
    """Merge-only: repo root .env (many users keep PGPASSWORD / DATABASE_URL here)."""
    dotenv_path = _repo_root() / ".env"
    if not dotenv_path.exists():
        return
    for line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        if key and _should_merge_dotenv_key(key):
            os.environ[key] = value


def load_pipeline_env() -> None:
    """
    Load ``agent/.env``, ``pipeline/.env``, and repo ``.env`` into ``os.environ``.
    Call this before reading ``TOURIST_APT_*`` or other pipeline env vars — not only on DB connect.
    """
    _load_agent_dotenv_override()
    _load_env_from_pipeline_dotenv()
    _load_env_from_repo_root_dotenv()


def get_pg_conn() -> Any:
    """
    Return a psycopg2 connection. Prefer ``DATABASE_URL`` when set (after loading dotenv).
    """
    load_pipeline_env()
    dsn = os.getenv("DATABASE_URL")
    if dsn:
        return psycopg2.connect(dsn)
    return psycopg2.connect(
        host=os.getenv("PGHOST", "localhost"),
        port=os.getenv("PGPORT", "5432"),
        user=os.getenv("PGUSER", "postgres"),
        password=os.getenv("PGPASSWORD", ""),
        dbname=os.getenv("PGDATABASE", "rooster"),
    )
