#!/usr/bin/env bash
# 1) pg_dump from LOCAL_DATABASE_URL → rooster.dump (custom -Fc)
# 2) DROP/CREATE the database named in RAILWAY_DATABASE_URL (empty target)
# 3) pg_restore into Railway
#
# Config: scripts/.env (override with ROOSTER_SYNC_ENV=/path/to/.env)
#
# Requires pg_dump / pg_restore / psql matching the *server* major version (e.g. PG 17).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENV_FILE="${ROOSTER_SYNC_ENV:-$ROOT/scripts/.env}"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Missing $ENV_FILE — copy scripts/.env.example to scripts/.env and set URLs." >&2
  exit 1
fi

set -a
# shellcheck source=/dev/null
source "$ENV_FILE"
set +a

# If a URL has no `user@` part, libpq uses the *current OS username* and prompts
# for the wrong role. Default to `postgres` unless PGUSER is set in .env.
: "${PGUSER:=postgres}"
export PGUSER

_trim_var() {
  local n="$1"
  local v="${!n:-}"
  v="${v//$'\r'/}"
  v="${v#"${v%%[![:space:]]*}"}"
  v="${v%"${v##*[![:space:]]}"}"
  printf -v "$n" '%s' "$v"
}

_append_sslmode_disable() {
  local n="$1"
  local u="${!n}"
  if [[ "$u" != *sslmode=* ]]; then
    if [[ "$u" == *\?* ]]; then
      printf -v "$n" '%s' "${u}&sslmode=disable"
    else
      printf -v "$n" '%s' "${u}?sslmode=disable"
    fi
  fi
}

_trim_var LOCAL_DATABASE_URL
_trim_var RAILWAY_DATABASE_URL

if [[ -z "${LOCAL_DATABASE_URL:-}" || -z "${RAILWAY_DATABASE_URL:-}" ]]; then
  echo "Set LOCAL_DATABASE_URL and RAILWAY_DATABASE_URL in $ENV_FILE" >&2
  exit 1
fi

_append_sslmode_disable RAILWAY_DATABASE_URL
export RAILWAY_DATABASE_URL

DUMP="${DUMP_FILE:-rooster.dump}"
if [[ "$DUMP" != /* ]]; then
  DUMP="$ROOT/$DUMP"
fi

echo "=== 1/3 pg_dump (local) → $DUMP ==="
pg_dump -U "$PGUSER" -Fc --no-owner --file="$DUMP" "$LOCAL_DATABASE_URL"

if ! pg_restore -l "$DUMP" 2>/dev/null | grep -qF "SCHEMA - core "; then
  echo "Warning: dump archive has no SCHEMA core — check LOCAL_DATABASE_URL points at your full dev DB." >&2
fi
if ! pg_restore -l "$DUMP" 2>/dev/null | grep -qF "SCHEMA - analytics "; then
  echo "Warning: dump archive has no SCHEMA analytics — run dbt locally first." >&2
fi

echo "=== 2/3 Wipe Railway database (DROP/CREATE) ==="
read -r TARGET_DB ADMIN_URL < <(RAILWAY_DATABASE_URL="$RAILWAY_DATABASE_URL" python3 -c "
from urllib.parse import urlparse, urlunparse
import os, sys
u = os.environ['RAILWAY_DATABASE_URL']
p = urlparse(u)
segs = [s for s in p.path.split('/') if s]
if not segs:
    print('RAILWAY_DATABASE_URL must end with a database name, e.g. .../railway', file=sys.stderr)
    sys.exit(1)
target = segs[-1]
segs[-1] = 'postgres'
admin = urlunparse(p._replace(path='/' + '/'.join(segs)))
print(target)
print(admin)
")

export ADMIN_URL
psql -U "$PGUSER" "$ADMIN_URL" -v ON_ERROR_STOP=1 -c \
  "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = '${TARGET_DB}' AND pid <> pg_backend_pid();" \
  || true
psql -U "$PGUSER" "$ADMIN_URL" -v ON_ERROR_STOP=1 -c "DROP DATABASE IF EXISTS \"${TARGET_DB}\";"
psql -U "$PGUSER" "$ADMIN_URL" -v ON_ERROR_STOP=1 -c "CREATE DATABASE \"${TARGET_DB}\";"

echo "=== 3/3 pg_restore → Railway ==="
pg_restore -U "$PGUSER" --no-owner --no-acl -d "$RAILWAY_DATABASE_URL" "$DUMP"

echo "Done. Database: $TARGET_DB on Railway."
