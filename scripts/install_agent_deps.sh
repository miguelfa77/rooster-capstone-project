#!/usr/bin/env bash
# Install agent/requirements.txt into the repo-root venv and verify imports.
# Run from anywhere:  bash scripts/install_agent_deps.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$ROOT/bin/python"
PIP="$ROOT/bin/pip"

if [[ ! -x "$PY" ]]; then
  echo "Expected venv at $ROOT/bin/python — create with: python3 -m venv . (or your usual setup)"
  exit 1
fi

echo "==> Upgrading pip (optional)…"
"$PIP" install --upgrade pip

echo "==> Reinstalling blinker 1.7.0 (last line with blinker._saferef; 1.8+ breaks selenium-wire / some stacks)…"
"$PIP" install --force-reinstall --no-cache-dir "blinker==1.7.0"

echo "==> Installing agent/requirements.txt (no cache — full wheels)…"
# Let this run to completion; resolver can take several minutes on slow networks.
"$PIP" install --no-cache-dir -r "$ROOT/agent/requirements.txt"

echo "==> Smoke test imports…"
"$PY" -c "import streamlit, folium, blinker as b; print('streamlit', streamlit.__version__, '| blinker', getattr(b, '__version__', '?'))"

echo "Done."
