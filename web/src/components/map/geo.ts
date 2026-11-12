// Pure helpers turning API rows into GeoJSON and MapLibre style expressions.
import type { Feature, FeatureCollection, Geometry, Position } from "geojson";
import type { ExpressionSpecification } from "maplibre-gl";
import { toNumber } from "@/lib/format";
import type { ChoroplethPayload } from "@/lib/types";

type Row = Record<string, unknown>;
type Colormap = ChoroplethPayload["colormap"];

// ColorBrewer 5-class sequential/diverging ramps (light → dark).
const RAMPS: Record<Colormap, string[]> = {
  YlOrRd: ["#ffffb2", "#fecc5c", "#fd8d3c", "#f03b20", "#bd0026"],
  YlGn: ["#ffffcc", "#c2e699", "#78c679", "#31a354", "#006837"],
  Blues: ["#eff3ff", "#bdd7e7", "#6baed6", "#3182bd", "#08519c"],
  RdYlGn: ["#d7191c", "#fdae61", "#ffffbf", "#a6d96a", "#1a9641"],
  Purples: ["#f2f0f7", "#cbc9e2", "#9e9ac8", "#756bb1", "#54278f"],
};

const GEOMETRY_TYPES = new Set(["Polygon", "MultiPolygon", "Point", "MultiPoint", "LineString", "MultiLineString"]);

function isGeometry(value: unknown): value is Geometry {
  if (typeof value !== "object" || value === null) return false;
  const g = value as { type?: unknown; coordinates?: unknown };
  return typeof g.type === "string" && GEOMETRY_TYPES.has(g.type) && Array.isArray(g.coordinates);
}

function without(row: Row, ...keys: string[]): Row {
  const out: Row = {};
  for (const [k, v] of Object.entries(row)) if (!keys.includes(k)) out[k] = v;
  return out;
}

/** Rows with a GeoJSON `geom` become features; the metric column is coerced to a number (or null). */
export function rowsToPolygons(rows: Row[], metric: string): FeatureCollection {
  const features: Feature[] = [];
  for (const row of rows) {
    if (!isGeometry(row.geom)) continue;
    const properties = without(row, "geom");
    if (metric in properties) properties[metric] = toNumber(properties[metric]);
    features.push({ type: "Feature", geometry: row.geom, properties });
  }
  return { type: "FeatureCollection", features };
}

/** Rows with finite lat/lng become Point features; coordinates are dropped from properties. */
export function rowsToPoints(rows: Row[], latKey = "lat", lngKey = "lng"): FeatureCollection {
  const features: Feature[] = [];
  for (const row of rows) {
    const lat = toNumber(row[latKey]);
    const lng = toNumber(row[lngKey]);
    if (lat === null || lng === null) continue;
    features.push({
      type: "Feature",
      geometry: { type: "Point", coordinates: [lng, lat] },
      properties: without(row, latKey, lngKey),
    });
  }
  return { type: "FeatureCollection", features };
}

export function colorRamp(name: Colormap): string[] {
  return RAMPS[name] ?? RAMPS.YlOrRd;
}

/** Linear interpolation of `field` across the ramp, with stops evenly spread over [vmin, vmax]. */
export function stepExpression(field: string, vmin: number, vmax: number, ramp: string[]): ExpressionSpecification {
  const span = vmax > vmin ? vmax - vmin : 1; // stops must be strictly ascending
  const last = Math.max(1, ramp.length - 1);
  const stops = ramp.flatMap((color, i) => [vmin + (span * i) / last, color]);
  return ["interpolate", ["linear"], ["get", field], ...stops] as ExpressionSpecification;
}

/** [min, max] of the numeric values of `field`, or null when there are none. */
export function extent(rows: Row[], field: string): [number, number] | null {
  let min = Infinity;
  let max = -Infinity;
  for (const row of rows) {
    const n = toNumber(row[field]);
    if (n === null) continue;
    if (n < min) min = n;
    if (n > max) max = n;
  }
  return min <= max ? [min, max] : null;
}

function visitPositions(coords: unknown, visit: (p: Position) => void): void {
  if (!Array.isArray(coords)) return;
  if (typeof coords[0] === "number") {
    visit(coords as Position);
    return;
  }
  for (const c of coords) visitPositions(c, visit);
}

/** [[west, south], [east, north]] covering every feature, or null for an empty collection. */
export function boundsOf(fc: FeatureCollection): [[number, number], [number, number]] | null {
  let w = Infinity;
  let s = Infinity;
  let e = -Infinity;
  let n = -Infinity;
  for (const f of fc.features) {
    const g = f.geometry;
    if (!g || g.type === "GeometryCollection") continue;
    visitPositions(g.coordinates, ([x, y]) => {
      if (x < w) w = x;
      if (x > e) e = x;
      if (y < s) s = y;
      if (y > n) n = y;
    });
  }
  return w <= e && s <= n ? [[w, s], [e, n]] : null;
}
