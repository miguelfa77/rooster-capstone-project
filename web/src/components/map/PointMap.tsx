"use client";

import { Popup, type ExpressionSpecification, type Map as MapLibreMap, type MapLayerMouseEvent } from "maplibre-gl";
import { useMemo } from "react";
import EmptyState from "@/components/primitives/EmptyState";
import { toNumber } from "@/lib/format";
import type { PointMapPayload } from "@/lib/types";
import BaseMap, { tokenColor } from "./BaseMap";
import { boundsOf, colorRamp, extent, rowsToPoints, stepExpression } from "./geo";
import Legend from "./Legend";
import { popupContent } from "./popup";

const SOURCE = "points";
const LAYER = "points";
const RAMP = colorRamp("YlOrRd");

export default function PointMap({ p }: { p: PointMapPayload }) {
  const colorField = p.color_field;
  const fc = useMemo(() => {
    const rows = colorField ? p.data.map((r) => ({ ...r, [colorField]: toNumber(r[colorField]) })) : p.data;
    return rowsToPoints(rows);
  }, [p.data, colorField]);
  const range = useMemo(
    () => (colorField ? extent(fc.features.map((f) => f.properties ?? {}), colorField) : null),
    [fc, colorField],
  );
  if (fc.features.length === 0) return <EmptyState />;

  const onLoad = (map: MapLibreMap) => {
    const accent = tokenColor("--primary");
    const color: ExpressionSpecification | string =
      colorField && range
        ? ["case", ["==", ["typeof", ["get", colorField]], "number"], stepExpression(colorField, range[0], range[1], RAMP), accent]
        : accent;
    map.addSource(SOURCE, { type: "geojson", data: fc });
    map.addLayer({
      id: LAYER,
      type: "circle",
      source: SOURCE,
      paint: { "circle-color": color, "circle-radius": 6, "circle-stroke-color": tokenColor("--card"), "circle-stroke-width": 1 },
    });

    map.on("click", LAYER, (e: MapLayerMouseEvent) => {
      const props = e.features?.[0]?.properties;
      if (!props) return;
      new Popup({ maxWidth: "260px" })
        .setLngLat(e.lngLat)
        .setDOMContent(popupContent(p.popup_fields ?? [], props, "url"))
        .addTo(map);
    });
    map.on("mouseenter", LAYER, () => (map.getCanvas().style.cursor = "pointer"));
    map.on("mouseleave", LAYER, () => (map.getCanvas().style.cursor = ""));

    const bounds = boundsOf(fc);
    if (bounds) map.fitBounds(bounds, { padding: 32, maxZoom: 15, duration: 0 });
  };

  return (
    <BaseMap onLoad={onLoad}>
      {range && <Legend title={p.color_label ?? colorField ?? ""} ramp={RAMP} range={range} />}
    </BaseMap>
  );
}
