"use client";

import type { FeatureCollection } from "geojson";
import {
  Popup,
  type ExpressionSpecification,
  type GeoJSONSource,
  type LayerSpecification,
  type Map as MapLibreMap,
  type MapLayerMouseEvent,
} from "maplibre-gl";
import { useEffect, useRef, useState } from "react";
import BaseMap from "@/components/map/BaseMap";
import { rowsToPoints, stepExpression } from "@/components/map/geo";
import Legend from "@/components/map/Legend";
import { popupContent } from "@/components/map/popup";
import { getIntel } from "@/lib/client";
import { CHOROPLETH_METRICS, COPY } from "@/lib/copy";
import { formatCell, toNumber } from "@/lib/format";
import type { ListingRow, NeighborhoodsOut } from "@/lib/types";
import styles from "./Intel.module.css";
import Section from "./Section";

type LayerKey = "barrios" | "parcels" | "vut" | "transit" | "listings";
/** What a layer draws: its features, a color range when it has one, and (barrios) the metric it was fetched for. */
type LayerData = { fc: FeatureCollection; range: [number, number] | null; metric?: string };

// Stacking order, bottom to top (toggles below keep the Streamlit order).
const Z_ORDER: LayerKey[] = ["barrios", "parcels", "vut", "transit", "listings"];
const TOGGLES: { key: LayerKey; label: string }[] = [
  { key: "barrios", label: COPY.toggleBarrios },
  { key: "transit", label: COPY.toggleTransport },
  { key: "vut", label: COPY.toggleVut },
  { key: "listings", label: COPY.toggleListings },
  { key: "parcels", label: COPY.toggleParcels },
];
const OFF: Record<LayerKey, boolean> = { barrios: false, parcels: false, vut: false, transit: false, listings: false };

// Colors as in the Streamlit explorer map.
const RAMP = ["#3B8BD4", "#1D9E75", "#EF9F27", "#D85A30"];
const NO_VALUE = "#d9d9d9";
const TRANSIT_METRO = "#534AB7";
const TRANSIT_BUS = "#1D9E75";
const TRANSIT_OTHER = "#888780";
const VUT = "#EF9F27";
const VENTA = "#E95D4F";
const ALQUILER = "#48A9DA";
const NO_OP = "#888888";
const PARCEL_LINE = "#1a5276";

const DEFAULT_METRIC = CHOROPLETH_METRICS[0].value;
const metricLabel = (metric: string) => CHOROPLETH_METRICS.find((m) => m.value === metric)?.label ?? metric;

function pathFor(key: LayerKey, metric: string): string {
  switch (key) {
    case "barrios":
      return `neighborhoods?metric=${encodeURIComponent(metric)}`;
    case "transit":
      return "overlays/transit";
    case "vut":
      return "overlays/tourism";
    case "listings":
      return "listings?operation=All";
    case "parcels":
      return "parcels";
  }
}

function asFeatureCollection(value: unknown): FeatureCollection {
  const fc = value as Partial<FeatureCollection> | null;
  if (!fc || fc.type !== "FeatureCollection" || !Array.isArray(fc.features)) throw new Error("not a FeatureCollection");
  return fc as FeatureCollection;
}

function asRows(value: unknown): Record<string, unknown>[] {
  if (!Array.isArray(value)) throw new Error("expected an array of rows");
  return value as Record<string, unknown>[];
}

/** Linear-interpolated quantile of sorted values (pandas' default). */
function quantile(sorted: number[], q: number): number {
  const pos = (sorted.length - 1) * q;
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
}

function toLayerData(key: LayerKey, raw: unknown, metric: string): LayerData {
  switch (key) {
    case "barrios": {
      const body = raw as NeighborhoodsOut;
      const range: [number, number] | null =
        body?.vmin !== null && body?.vmin !== undefined && body.vmax !== null && body.vmax !== undefined
          ? [body.vmin, body.vmax]
          : null;
      return { fc: asFeatureCollection(body?.geojson), range, metric };
    }
    case "parcels":
      return { fc: asFeatureCollection(raw), range: null };
    case "listings": {
      const rows = asRows(raw).map((r: ListingRow) => ({
        ...r,
        eur_per_sqm: toNumber(r.eur_per_sqm),
        op: String(r.operation ?? "").toLowerCase(),
      }));
      // 5th–95th percentile of €/m², so a few outliers do not wash out the fill colors.
      const values = rows.map((r) => r.eur_per_sqm).filter((n): n is number => n !== null).sort((a, b) => a - b);
      const range: [number, number] | null = values.length ? [quantile(values, 0.05), quantile(values, 0.95)] : null;
      return { fc: rowsToPoints(rows), range };
    }
    default:
      return { fc: rowsToPoints(asRows(raw)), range: null };
  }
}

const isNumber = (field: string): ExpressionSpecification => ["==", ["typeof", ["get", field]], "number"];

function barriosColor(range: [number, number] | null): ExpressionSpecification | string {
  return range ? ["case", isNumber("value"), stepExpression("value", range[0], range[1], RAMP), NO_VALUE] : NO_VALUE;
}

function listingsColor(range: [number, number] | null): ExpressionSpecification | string {
  return range ? ["case", isNumber("eur_per_sqm"), stepExpression("eur_per_sqm", range[0], range[1], RAMP), RAMP[0]] : RAMP[0];
}

const sourceId = (key: LayerKey) => `intel-${key}`;

/** Layer specs per overlay; the first layer of each is the one that opens popups. */
function layerSpecs(key: LayerKey, d: LayerData): LayerSpecification[] {
  const source = sourceId(key);
  switch (key) {
    case "barrios":
      return [
        {
          id: "intel-barrios-fill",
          type: "fill",
          source,
          paint: {
            "fill-color": barriosColor(d.range),
            "fill-opacity": ["case", isNumber("value"), 0.72, 0.35],
          },
        },
        { id: "intel-barrios-line", type: "line", source, paint: { "line-color": "#333333", "line-width": 0.5 } },
      ];
    case "parcels":
      // Thousands of polygons: outlines only, thin and translucent.
      return [
        { id: "intel-parcels", type: "line", source, paint: { "line-color": PARCEL_LINE, "line-width": 0.35, "line-opacity": 0.55 } },
      ];
    case "vut":
      return [
        {
          id: "intel-vut",
          type: "circle",
          source,
          paint: { "circle-color": VUT, "circle-radius": 4, "circle-opacity": 0.65, "circle-stroke-color": VUT, "circle-stroke-width": 1 },
        },
      ];
    case "transit": {
      const kind: ExpressionSpecification = ["downcase", ["to-string", ["coalesce", ["get", "stop_type"], ""]]];
      const color: ExpressionSpecification = [
        "match",
        kind,
        ["station", "halt", "tram_stop"],
        TRANSIT_METRO,
        "stop_position",
        TRANSIT_BUS,
        TRANSIT_OTHER,
      ];
      return [
        {
          id: "intel-transit",
          type: "circle",
          source,
          paint: { "circle-color": color, "circle-radius": 5, "circle-opacity": 0.85, "circle-stroke-color": color, "circle-stroke-width": 1 },
        },
      ];
    }
    case "listings":
      return [
        {
          id: "intel-listings",
          type: "circle",
          source,
          paint: {
            "circle-color": listingsColor(d.range),
            "circle-opacity": 0.82,
            "circle-radius": 4,
            "circle-stroke-color": ["match", ["get", "op"], "venta", VENTA, "alquiler", ALQUILER, NO_OP],
            "circle-stroke-width": 2.5,
          },
        },
      ];
  }
}

const LAYER_IDS: Record<LayerKey, string[]> = {
  barrios: ["intel-barrios-fill", "intel-barrios-line"],
  parcels: ["intel-parcels"],
  vut: ["intel-vut"],
  transit: ["intel-transit"],
  listings: ["intel-listings"],
};

function lines(...texts: string[]): HTMLElement {
  const root = document.createElement("div");
  for (const t of texts) {
    if (!t) continue;
    const line = document.createElement("div");
    line.textContent = t;
    root.appendChild(line);
  }
  return root;
}

function popupFor(key: LayerKey, props: Record<string, unknown>, metric: string): HTMLElement {
  const text = (v: unknown) => (v === null || v === undefined ? "" : String(v));
  switch (key) {
    case "barrios":
      return popupContent(
        [
          { field: "name", label: COPY.tooltipNeighborhood },
          { field: "value", label: metricLabel(metric) },
          { field: "listings", label: COPY.toggleListings },
          { field: "yield_pct", label: COPY.tooltipYield },
        ],
        props,
      );
    case "transit":
      return lines(text(props.name) || "—", text(props.stop_type));
    case "vut":
      return lines(text(props.address).slice(0, 60));
    case "listings": {
      const op = props.op === "venta" ? COPY.popupVenta : props.op === "alquiler" ? COPY.popupAlquiler : "—";
      const area = toNumber(props.area_sqm);
      const root = popupContent([], props, "url");
      root.prepend(
        lines(`${op} · ${formatCell(props.price_int, "currency")} · ${area === null ? "—" : formatCell(area, "integer")} m²`),
      );
      return root;
    }
    case "parcels":
      return lines();
  }
}

/** Adds the overlay's source and layers on first use; afterwards only swaps data and color ramps. */
function applyData(map: MapLibreMap, key: LayerKey, d: LayerData, popup: Popup, metricOf: () => string) {
  const existing = map.getSource(sourceId(key)) as GeoJSONSource | undefined;
  if (existing) {
    existing.setData(d.fc);
    if (key === "barrios") map.setPaintProperty("intel-barrios-fill", "fill-color", barriosColor(d.range));
    if (key === "listings") map.setPaintProperty("intel-listings", "circle-color", listingsColor(d.range));
    return;
  }
  map.addSource(sourceId(key), { type: "geojson", data: d.fc });
  // Keep the stacking order no matter which toggle was turned on first.
  const above = Z_ORDER.slice(Z_ORDER.indexOf(key) + 1).flatMap((k) => LAYER_IDS[k]);
  const beforeId = above.find((id) => map.getLayer(id));
  const specs = layerSpecs(key, d);
  for (const spec of specs) map.addLayer(spec, beforeId);
  if (key === "parcels") return;
  const hit = specs[0].id;
  map.on("click", hit, (e: MapLayerMouseEvent) => {
    const props = e.features?.[0]?.properties;
    if (props) popup.setLngLat(e.lngLat).setDOMContent(popupFor(key, props, metricOf())).addTo(map);
  });
  map.on("mouseenter", hit, () => (map.getCanvas().style.cursor = "pointer"));
  map.on("mouseleave", hit, () => (map.getCanvas().style.cursor = ""));
}

/** Section 2: Valencia map with five overlays, each fetched the first time it is turned on. */
export default function IntelMap() {
  const [map, setMap] = useState<MapLibreMap | null>(null);
  const [metric, setMetric] = useState(DEFAULT_METRIC);
  const [on, setOn] = useState(OFF);
  const [failed, setFailed] = useState(OFF);
  const [data, setData] = useState<Partial<Record<LayerKey, LayerData>>>({});
  const pending = useRef<Partial<Record<LayerKey, string>>>({});
  // Datasets already pushed to the current map instance (reset if BaseMap ever hands us a new one).
  const applied = useRef<{ map: MapLibreMap | null; data: Partial<Record<LayerKey, LayerData>> }>({ map: null, data: {} });
  const popup = useRef<Popup | null>(null);
  const barriosMetric = useRef(DEFAULT_METRIC);

  const ready = (key: LayerKey) => {
    const d = data[key];
    return !!d && (key !== "barrios" || d.metric === metric);
  };

  const load = (key: LayerKey, m: string) => {
    const path = pathFor(key, m);
    if (pending.current[key] === path) return;
    pending.current[key] = path;
    getIntel<unknown>(path)
      .then((raw) => toLayerData(key, raw, m))
      .then(
        (d) => {
          if (pending.current[key] !== path) return; // superseded by a newer request
          pending.current[key] = undefined;
          setData((prev) => ({ ...prev, [key]: d }));
        },
        (err: unknown) => {
          if (pending.current[key] !== path) return;
          pending.current[key] = undefined;
          console.warn(`intel layer ${key} failed`, err);
          setFailed((f) => ({ ...f, [key]: true }));
          setOn((o) => ({ ...o, [key]: false }));
        },
      );
  };

  const toggle = (key: LayerKey, checked: boolean) => {
    setOn((o) => ({ ...o, [key]: checked }));
    if (!checked) return;
    setFailed((f) => ({ ...f, [key]: false }));
    if (!ready(key)) load(key, metric);
  };

  const changeMetric = (m: string) => {
    setMetric(m);
    if (data.barrios?.metric === m) pending.current.barrios = undefined; // drop any in-flight other metric
    else if (on.barrios) load("barrios", m);
  };

  // Mirror React state onto the map: add/update sources once per new dataset, then set visibility.
  useEffect(() => {
    if (!map) return;
    popup.current ??= new Popup({ maxWidth: "260px" });
    if (applied.current.map !== map) applied.current = { map, data: {} };
    for (const key of Z_ORDER) {
      const d = data[key];
      if (d && applied.current.data[key] !== d) {
        applyData(map, key, d, popup.current, () => barriosMetric.current);
        applied.current.data[key] = d;
        if (key === "barrios" && d.metric) barriosMetric.current = d.metric;
      }
      const visible = on[key] && !!d && (key !== "barrios" || d.metric === metric);
      for (const id of LAYER_IDS[key]) {
        if (map.getLayer(id)) map.setLayoutProperty(id, "visibility", visible ? "visible" : "none");
      }
    }
  }, [map, data, on, metric]);

  const barrios = on.barrios && ready("barrios") ? data.barrios : undefined;
  const loading = Z_ORDER.some((k) => on[k] && !ready(k));

  return (
    <Section title={COPY.sectionMapa}>
      <label className={styles.select}>
        {COPY.intelMapColorBy}
        <select value={metric} onChange={(e) => changeMetric(e.target.value)}>
          {CHOROPLETH_METRICS.map((m) => (
            <option key={m.value} value={m.value}>
              {m.label}
            </option>
          ))}
        </select>
      </label>
      <div className={styles.controls}>
        {TOGGLES.map(({ key, label }) => (
          <label key={key} className={styles.toggle}>
            <input type="checkbox" checked={on[key]} onChange={(e) => toggle(key, e.target.checked)} />
            {label}
          </label>
        ))}
      </div>
      {TOGGLES.filter(({ key }) => failed[key]).map(({ key, label }) => (
        <p key={key} role="alert" className={styles.error}>
          {label}: {COPY.mapLoadError}
        </p>
      ))}
      {loading ? <p className={styles.status}>{COPY.intelLoading}</p> : null}
      <BaseMap onLoad={setMap} height={480}>
        {barrios?.range ? <Legend title={metricLabel(metric)} ramp={RAMP} range={barrios.range} /> : null}
      </BaseMap>
      {barrios && metric === "investment_score" ? <p className={styles.caption}>{COPY.mapLegendInvestment}</p> : null}
      {on.listings && ready("listings") ? <p className={styles.caption}>{COPY.intelListingsOpLegend}</p> : null}
    </Section>
  );
}
