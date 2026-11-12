"use client";

import {
  Popup,
  type ExpressionSpecification,
  type LngLat,
  type Map as MapLibreMap,
  type MapLayerMouseEvent,
  type MapMouseEvent,
} from "maplibre-gl";
import { useMemo } from "react";
import EmptyState from "@/components/primitives/EmptyState";
import type { ChoroplethPayload } from "@/lib/types";
import BaseMap from "./BaseMap";
import { boundsOf, colorRamp, extent, rowsToPolygons, stepExpression } from "./geo";
import Legend from "./Legend";
import { popupContent } from "./popup";

const SOURCE = "choropleth";
const FILL = "choropleth-fill";
const NO_DATA = "#2a2a2a";

export default function ChoroplethMap({ p }: { p: ChoroplethPayload }) {
  const fc = useMemo(() => rowsToPolygons(p.data, p.metric), [p.data, p.metric]);
  const range = useMemo(
    () => extent(fc.features.map((f) => f.properties ?? {}), p.metric),
    [fc, p.metric],
  );
  if (fc.features.length === 0) return <EmptyState />;

  const ramp = colorRamp(p.colormap);
  const onLoad = (map: MapLibreMap) => {
    const fillColor: ExpressionSpecification | string = range
      ? ["case", ["==", ["typeof", ["get", p.metric]], "number"], stepExpression(p.metric, range[0], range[1], ramp), NO_DATA]
      : NO_DATA;
    map.addSource(SOURCE, { type: "geojson", data: fc });
    map.addLayer({ id: FILL, type: "fill", source: SOURCE, paint: { "fill-color": fillColor, "fill-opacity": 0.7 } });
    map.addLayer({ id: "choropleth-outline", type: "line", source: SOURCE, paint: { "line-color": "#333", "line-width": 1 } });

    const popup = new Popup({ closeButton: false, closeOnClick: false });
    const show = (lngLat: LngLat, props: Record<string, unknown> | undefined) => {
      if (!props) return;
      popup.setLngLat(lngLat).setDOMContent(popupContent(p.tooltip_fields ?? [], props)).addTo(map);
    };
    // Desktop: hover. Touch: tap a polygon to open, tap anywhere else to close.
    map.on("mousemove", FILL, (e: MapLayerMouseEvent) => show(e.lngLat, e.features?.[0]?.properties));
    map.on("mouseleave", FILL, () => popup.remove());
    map.on("click", (e: MapMouseEvent) => {
      const hit = map.queryRenderedFeatures(e.point, { layers: [FILL] })[0];
      if (hit) show(e.lngLat, hit.properties);
      else popup.remove();
    });

    const bounds = boundsOf(fc);
    if (bounds) map.fitBounds(bounds, { padding: 24, maxZoom: 15, duration: 0 });
  };

  return (
    <BaseMap onLoad={onLoad}>
      {range && <Legend title={p.metric_label} ramp={ramp} range={range} />}
    </BaseMap>
  );
}
