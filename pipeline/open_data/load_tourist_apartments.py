"""
Load tourist apartment licenses from GVA open data (CSV URL or local path).

The Comunitat Valenciana registry has no coordinates — we join ``ref_catastral`` to
``core.parcels.refcat`` and use parcel centroid (25830 → 4326). Optionally geocode
unmatched rows via Nominatim (slow; off by default).

Env:
  TOURIST_APT_CSV_URL   — optional HTTP(S) CSV
  TOURIST_APT_CSV_PATH  — local CSV if URL unset or download fails
  TOURIST_APT_PROVINCE_CODES — comma-separated INE province codes to keep (default: 46 = Valencia).
                             Example: ``46`` for all municipios in provincia Valencia only.
  TOURIST_APT_GEOCODE_UNMATCHED — if ``1``, Nominatim geocode rows with no parcel match (rate-limited).

Run:
  python -m pipeline.open_data.load_tourist_apartments
"""

from __future__ import annotations

import io
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.raw.db_utils import get_pg_conn, load_pipeline_env


def _norm_col(c: str) -> str:
    return re.sub(r"\s+", " ", str(c).strip().lower())


def _pick_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    lower = {_norm_col(c): c for c in df.columns}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    return None


def normalize_status(val: object) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "unknown"
    s = str(val).strip().lower()
    if any(x in s for x in ("activ", "alta", "vigent", "valid", "active")):
        return "active"
    if any(x in s for x in ("baixa", "cancel", "revoc", "suspens", "inactiv")):
        return "inactive"
    return s[:80] if s else "unknown"


def _norm_province_code(val: object) -> str:
    """INE-style province code as zero-padded 2 digits (03, 12, 46)."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    s = str(val).strip()
    if not s or s.lower() == "nan":
        return ""
    if re.match(r"^\d+$", s):
        return f"{int(s):02d}"
    return s.upper()


def _parse_province_codes() -> set[str]:
    raw = (os.getenv("TOURIST_APT_PROVINCE_CODES") or "46").strip()
    out: set[str] = set()
    for part in raw.split(","):
        p = part.strip()
        if not p:
            continue
        if p.isdigit():
            out.add(f"{int(p):02d}")
        else:
            out.add(p.upper())
    return out or {"46"}


def _detect_sep(sample: bytes) -> str:
    try:
        head = sample[:8000].decode("utf-8", errors="replace")
    except Exception:
        return ","
    first = head.splitlines()[0] if head else ""
    return ";" if first.count(";") >= first.count(",") else ","


def load_dataframe() -> pd.DataFrame:
    url = (os.getenv("TOURIST_APT_CSV_URL") or "").strip()
    path = (os.getenv("TOURIST_APT_CSV_PATH") or "").strip()

    raw: bytes | None = None
    if url:
        try:
            print(f"Downloading {url[:90]}…")
            req = urllib.request.Request(url, headers={"User-Agent": "RoosterOpenData/1.0"})
            with urllib.request.urlopen(req, timeout=300) as resp:
                raw = resp.read()
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            print(f"URL load failed ({e}); trying file path…", file=sys.stderr)

    if raw is None:
        if not path:
            print(
                "Set TOURIST_APT_CSV_URL or TOURIST_APT_CSV_PATH (see README).",
                file=sys.stderr,
            )
            sys.exit(1)
        p = Path(path)
        if not p.is_file():
            print(f"File not found: {p}", file=sys.stderr)
            sys.exit(1)
        raw = p.read_bytes()

    sep = _detect_sep(raw)
    print(f"Detected CSV separator: {sep!r}")
    return pd.read_csv(
        io.BytesIO(raw),
        sep=sep,
        engine="python",
        encoding="utf-8",
        on_bad_lines="skip",
        dtype=str,
    )


def normalize_refcat(val: object) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    s = re.sub(r"\s+", "", str(val).strip().upper())
    return s


def filter_by_province(df: pd.DataFrame, codes: set[str]) -> pd.DataFrame:
    """Keep rows whose INE provincia code matches (e.g. 46 = Valencia province, all municipios)."""
    col_cod = _pick_column(df, ("cod_provincia", "codigo provincia", "código provincia", "cod provincia"))
    if col_cod is not None:
        norm = df[col_cod].map(_norm_province_code)
        mask = norm.isin(codes)
        n = int(mask.sum())
        print(f"Province filter (cod_provincia in {sorted(codes)}): {n} rows (of {len(df)})")
        return df.loc[mask].copy()

    col_name = _pick_column(df, ("provincia",))
    if col_name is not None:
        # Match province name: VALÈNCIA/VALENCIA as provincia (not city listings elsewhere)
        s = df[col_name].astype(str)
        mask = s.str.contains(
            r"VALÈNCIA/VALENCIA|VALENCIA/VALÈNCIA",
            case=False,
            regex=True,
            na=False,
        )
        n = int(mask.sum())
        print(f"Province filter (provincia name ~ Valencia province): {n} rows (of {len(df)})")
        return df.loc[mask].copy()

    print(
        "Warning: no cod_provincia or provincia column — not filtering by province.",
        file=sys.stderr,
    )
    return df


def fetch_refcat_centroids(cur, refcats: list[str]) -> dict[str, tuple[float, float]]:
    """Map refcat -> (lng, lat) WGS84 from parcel polygon centroids."""
    out: dict[str, tuple[float, float]] = {}
    if not refcats:
        return out
    batch = 4000
    for i in range(0, len(refcats), batch):
        chunk = [r for r in refcats[i : i + batch] if r]
        if not chunk:
            continue
        cur.execute(
            """
            SELECT refcat,
                   ST_X(ST_Transform(ST_Centroid(geom::geometry), 4326)) AS lng,
                   ST_Y(ST_Transform(ST_Centroid(geom::geometry), 4326)) AS lat
            FROM core.parcels
            WHERE refcat = ANY(%s)
            """,
            (chunk,),
        )
        for refcat, lng, lat in cur.fetchall():
            if lng is not None and lat is not None:
                out[normalize_refcat(refcat)] = (float(lng), float(lat))
    return out


def geocode_nominatim(address: str, municipio: str) -> tuple[float, float] | None:
    """Single Nominatim lookup (respect ToS: max ~1 req/s)."""
    q = f"{address}, {municipio}, Comunitat Valenciana, Spain"
    params = urllib.parse.urlencode({"q": q, "format": "json", "limit": 1, "countrycodes": "es"})
    url = f"https://nominatim.openstreetmap.org/search?{params}"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "RoosterCapstone/1.0 (tourist apt loader)"},
    )
    time.sleep(1.15)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, json.JSONDecodeError, OSError):
        return None
    if not data:
        return None
    lat = float(data[0]["lat"])
    lon = float(data[0]["lon"])
    return (lon, lat)


def main() -> None:
    # Must run before os.getenv — pipeline/.env is otherwise only loaded inside get_pg_conn().
    load_pipeline_env()

    province_codes = _parse_province_codes()
    print(f"Province codes (INE): {sorted(province_codes)}")

    df = load_dataframe()
    df.columns = [_norm_col(c) for c in df.columns]
    df = filter_by_province(df, province_codes)
    if df.empty:
        print("No rows left after province filter; nothing to load.", file=sys.stderr)
        sys.exit(0)

    col_id = _pick_column(
        df,
        ("signatura", "expediente", "id", "código", "codigo", "referencia", "numero"),
    )
    col_lic = _pick_column(df, ("numero licencia", "licencia", "license_no", "license", "num_licencia"))
    col_addr = _pick_column(df, ("direccion", "dirección", "address", "domicilio", "via"))
    col_ref = _pick_column(
        df,
        ("ref_catastral", "referencia catastral", "refcat", "referencia_catastral"),
    )
    col_muni = _pick_column(df, ("municipio", "municipi", "nombre municipio"))
    col_status = _pick_column(df, ("estado", "situacion", "situación", "status"))
    # Optional lat/lng if present in some exports
    col_lat = _pick_column(df, ("latitud", "lat", "ycoord", "coord_y"))
    col_lng = _pick_column(df, ("longitud", "lon", "lng", "xcoord", "coord_x"))

    geocode_unmatched = os.getenv("TOURIST_APT_GEOCODE_UNMATCHED", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )

    pending: list[dict] = []
    refcats_to_fetch: list[str] = []

    for _, row in df.iterrows():
        rid = ""
        if col_id:
            rid = str(row.get(col_id, "")).strip()
        lic = str(row.get(col_lic, "")).strip() if col_lic else ""
        addr = str(row.get(col_addr, "")).strip() if col_addr else ""
        muni = str(row.get(col_muni, "")).strip() if col_muni else ""
        ref = normalize_refcat(row.get(col_ref, "")) if col_ref else ""

        st_raw = row.get(col_status) if col_status else None
        if st_raw is None or (isinstance(st_raw, float) and pd.isna(st_raw)) or str(st_raw).strip() == "":
            status = "active"
        else:
            status = normalize_status(st_raw)

        if not rid:
            rid = lic or ref or f"row_{hash((addr, muni)) % 10**10}"
        rid = rid[:500]

        lat = lng = None
        if col_lat and col_lng:
            lat = pd.to_numeric(row.get(col_lat), errors="coerce")
            lng = pd.to_numeric(row.get(col_lng), errors="coerce")
            if pd.isna(lat) or pd.isna(lng):
                lat = lng = None
            else:
                lat, lng = float(lat), float(lng)

        if ref:
            refcats_to_fetch.append(ref)

        pending.append(
            {
                "id": rid,
                "license": lic[:200],
                "address": addr[:2000],
                "municipio": muni,
                "ref": ref,
                "status": status,
                "lat_csv": lat,
                "lng_csv": lng,
            }
        )

    conn = get_pg_conn()
    try:
        cur = conn.cursor()
        uniq_refs = list(dict.fromkeys(refcats_to_fetch))
        centroid_map = fetch_refcat_centroids(cur, uniq_refs)
        print(
            f"Parcel refcat matches: {len(centroid_map)} "
            f"(unique refcats in CSV: {len(uniq_refs)})"
        )

        matched = 0
        geocoded = 0
        skipped = 0

        for item in pending:
            lat, lng = item["lat_csv"], item["lng_csv"]

            if lat is None and item["ref"] and item["ref"] in centroid_map:
                lng, lat = centroid_map[item["ref"]]
                matched += 1
            elif lat is None and geocode_unmatched and item["address"]:
                g = geocode_nominatim(item["address"], item["municipio"] or "València")
                if g:
                    lng, lat = g
                    geocoded += 1
                else:
                    skipped += 1
            elif lat is None:
                skipped += 1

            item["_lat"], item["_lng"] = lat, lng

        print(
            f"Coordinates: parcel join={matched}, nominatim={geocoded}, "
            f"missing coords={skipped} (core requires lat/lng; active licenses only)"
        )

        records_raw = [
            (
                item["id"],
                item["address"],
                item["license"],
                item["_lat"],
                item["_lng"],
                item["status"],
            )
            for item in pending
        ]

        core_rows: list[tuple] = []
        for item in pending:
            lat, lng = item.get("_lat"), item.get("_lng")
            if lat is None or lng is None:
                continue
            if item["status"] != "active":
                continue
            core_rows.append(
                (
                    item["id"],
                    item["address"],
                    item["license"],
                    float(lat),
                    float(lng),
                    item["status"],
                    float(lng),
                    float(lat),
                )
            )

        cur.execute("TRUNCATE raw.tourist_apartments;")
        cur.execute("DELETE FROM core.tourist_apartments WHERE TRUE;")

        if records_raw:
            cur.executemany(
                """
                INSERT INTO raw.tourist_apartments (id, address, license_no, lat, lng, status)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                records_raw,
            )

        if core_rows:
            cur.executemany(
                """
                INSERT INTO core.tourist_apartments (
                    id, address, license_no, lat, lng, status, geom, neighborhood_id
                )
                VALUES (
                    %s, %s, %s, %s, %s, %s,
                    ST_SetSRID(ST_MakePoint(%s, %s), 4326),
                    NULL
                )
                """,
                core_rows,
            )

            cur.execute(
                """
                UPDATE core.tourist_apartments t
                SET neighborhood_id = (
                    SELECT n.id
                    FROM core.neighborhoods n
                    WHERE ST_Within(t.geom, n.geom)
                    LIMIT 1
                );
                """
            )

        conn.commit()
        cur.close()
        print(
            f"Done. raw.tourist_apartments={len(records_raw)} rows, "
            f"core.tourist_apartments={len(core_rows)} rows."
        )
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    main()
