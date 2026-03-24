#!/usr/bin/env bash
# Restore a pg_dump custom-format file to Railway Postgres.
#
# Preferred: set DATABASE_URL in scripts/railway-restore.env (paste from Railway Postgres
# Variables, or use the same ${{...}} value the web service uses). That avoids mismatches
# between POSTGRES_PASSWORD in the UI and the password actually stored in Postgres.
#
# Alternative: PGHOST, PGPORT, PGUSER, PGPASSWORD, PGDATABASE (and optional PGPASSWORD_FILE).
#
#   bash scripts/railway_pg_restore.sh test
#   bash scripts/railway_pg_restore.sh wipe   # DROP/CREATE target DB (needs DATABASE_URL)
#   bash scripts/railway_pg_restore.sh
#
# Optional: PG_RESTORE_CLEAN=1 adds --clean --if-exists (can fail on PostGIS if extensions
# depend on each other). For a fresh empty DB, leave unset (default).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENV_FILE="${RAILWAY_RESTORE_ENV:-$ROOT/scripts/railway-restore.env}"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Missing $ENV_FILE"
  echo "Copy scripts/railway-restore.env.example to scripts/railway-restore.env and fill in Railway values."
  exit 1
fi

set -a
# shellcheck source=/dev/null
source "$ENV_FILE"
set +a

_trim_var() {
  local n="$1"
  local v="${!n:-}"
  v="${v//$'\r'/}"
  v="${v#"${v%%[![:space:]]*}"}"
  v="${v%"${v##*[![:space:]]}"}"
  printf -v "$n" '%s' "$v"
}

_trim_var DATABASE_URL

# Optional: password from one-line file (only used with discrete PG* vars)
if [[ -z "${DATABASE_URL:-}" && -n "${PGPASSWORD_FILE:-}" ]]; then
  PF="$PGPASSWORD_FILE"
  [[ "$PF" == /* ]] || PF="$ROOT/$PF"
  if [[ -f "$PF" ]]; then
    IFS= read -r PGPASSWORD <"$PF" || true
    _trim_var PGPASSWORD
  else
    echo "Error: PGPASSWORD_FILE not found: $PF"
    exit 1
  fi
fi

if [[ -n "${DATABASE_URL:-}" ]]; then
  # Public TCP proxy is usually non-TLS; append sslmode=disable if not specified (fixes proxy + require mismatch).
  if [[ "$DATABASE_URL" != *sslmode=* ]]; then
    if [[ "$DATABASE_URL" == *\?* ]]; then
      DATABASE_URL="${DATABASE_URL}&sslmode=disable"
    else
      DATABASE_URL="${DATABASE_URL}?sslmode=disable"
    fi
  fi
  export DATABASE_URL
  CONN_DESC="DATABASE_URL"
else
  for _k in PGHOST PGPORT PGUSER PGDATABASE PGPASSWORD PGSSLMODE; do
    _trim_var "$_k"
  done
  for key in PGHOST PGPORT PGUSER PGDATABASE PGPASSWORD; do
    if [[ -z "${!key:-}" ]]; then
      echo "Error: set DATABASE_URL in $ENV_FILE, or set $key (and all of PGHOST PGPORT PGUSER PGDATABASE PGPASSWORD)."
      exit 1
    fi
  done
  export PGHOST PGPORT PGUSER PGDATABASE PGPASSWORD
  export PGSSLMODE="${PGSSLMODE:-disable}"
  CONN_DESC="$PGUSER@$PGHOST:$PGPORT/$PGDATABASE (sslmode=$PGSSLMODE)"
fi

DUMP_PATH="${DUMP_FILE:-rooster.dump}"
if [[ "$DUMP_PATH" != /* ]]; then
  DUMP_PATH="$ROOT/$DUMP_PATH"
fi
if [[ ! -f "$DUMP_PATH" ]]; then
  echo "Error: dump file not found: $DUMP_PATH"
  exit 1
fi

_run_psql() {
  if [[ -n "${DATABASE_URL:-}" ]]; then
    psql "$DATABASE_URL" "$@"
  else
    psql "$@"
  fi
}

_run_restore() {
  if [[ "${PG_RESTORE_CLEAN:-}" == "1" ]]; then
    if [[ -n "${DATABASE_URL:-}" ]]; then
      pg_restore --no-owner --no-acl --clean --if-exists -d "$DATABASE_URL" "$DUMP_PATH"
    else
      pg_restore --no-owner --no-acl --clean --if-exists \
        -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" -d "$PGDATABASE" \
        "$DUMP_PATH"
    fi
  else
    if [[ -n "${DATABASE_URL:-}" ]]; then
      pg_restore --no-owner --no-acl -d "$DATABASE_URL" "$DUMP_PATH"
    else
      pg_restore --no-owner --no-acl \
        -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" -d "$PGDATABASE" \
        "$DUMP_PATH"
    fi
  fi
}

if [[ "${1:-}" == "test" ]]; then
  echo "Testing connection (SELECT 1) via $CONN_DESC ..."
  _run_psql -v ON_ERROR_STOP=1 -c "SELECT 1 AS ok;"
  echo "OK"
  exit 0
fi

if [[ "${1:-}" == "wipe" ]]; then
  if [[ -z "${DATABASE_URL:-}" ]]; then
    echo "Error: wipe requires DATABASE_URL in $ENV_FILE"
    exit 1
  fi
  TARGET_DB="${PGDATABASE:-railway}"
  echo "Wiping database \"$TARGET_DB\" (connecting to postgres maintenance DB)..."
  ADMIN_URL="$(python3 -c "
from urllib.parse import urlparse, urlunparse
import os
u = os.environ['DATABASE_URL']
p = urlparse(u)
parts = [x for x in p.path.split('/') if x]
if not parts:
    raise SystemExit('bad DATABASE_URL path')
parts[-1] = 'postgres'
np = '/' + '/'.join(parts)
print(urlunparse(p._replace(path=np)))
")"
  export ADMIN_URL
  psql "$ADMIN_URL" -v ON_ERROR_STOP=1 -c \
    "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = '${TARGET_DB}' AND pid <> pg_backend_pid();"
  psql "$ADMIN_URL" -v ON_ERROR_STOP=1 -c "DROP DATABASE IF EXISTS \"${TARGET_DB}\";"
  psql "$ADMIN_URL" -v ON_ERROR_STOP=1 -c "CREATE DATABASE \"${TARGET_DB}\";"
  echo "Empty database \"${TARGET_DB}\" ready. Run: bash scripts/railway_pg_restore.sh"
  exit 0
fi

echo "Restoring from $DUMP_PATH using $CONN_DESC ..."
_run_restore

echo "Done."
