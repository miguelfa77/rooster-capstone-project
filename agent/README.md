# Agent (LLM layer)

Natural language → SQL over the Rooster PostgreSQL database.

## Setup

1. **Dependencies** (from repo root, with venv activated or using `./bin/pip`):

   Let the install **finish** (large wheels; pip resolver can sit a while):

   ```bash
   ./bin/pip install --upgrade pip
   ./bin/pip install --no-cache-dir -r agent/requirements.txt
   ```

   Or one shot:

   ```bash
   bash scripts/install_agent_deps.sh
   ```

   **`blinker._saferef` errors:** `blinker==1.7.0` is pinned (`_saferef` was removed in **blinker 1.8+**; selenium-wire and similar imports still need it). Fix a broken venv with:
   ```bash
   ./bin/pip install --force-reinstall --no-cache-dir "blinker==1.7.0"
   ./bin/pip install -r agent/requirements.txt
   ```
   **Flask 3** in the same venv conflicts (it wants `blinker>=1.9`): use a Rooster-only venv or `pip uninstall flask`.

2. **Config:**
   - `agent/.env` – `OPENAI_API_KEY` (or `OPENAI_KEY`) for all LLM calls (SQL layer, Ask Rooster agent)
   - `pipeline/.env` – `PGHOST`, `PGPORT`, `PGUSER`, `PGPASSWORD`, `PGDATABASE`

## Usage

**Streamlit UI** (from repo root):
```bash
./bin/python -m streamlit run app.py
```
Or: `./run_app.sh`

**Programmatic:**
```python
from agent.llm_sql import query
result = query("What are the top 5 neighborhoods by rental count?")
# result["sql"], result["rows"], result["error"]
```

## Data loading

Raw and core loaders live in `pipeline/raw/` and `pipeline/core/`. See `pipeline/raw/README.md`.
