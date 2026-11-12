"use client";

import { Map as MapLibreMap, NavigationControl, type StyleSpecification } from "maplibre-gl";
import { useEffect, useRef, useState, type ReactNode } from "react";
import EmptyState from "@/components/primitives/EmptyState";
import { COPY } from "@/lib/copy";
import styles from "./Map.module.css";

export const VALENCIA_CENTER: [number, number] = [-0.3763, 39.4699];
export const VALENCIA_ZOOM = 12;

const BASEMAP: StyleSpecification = {
  version: 8,
  sources: {
    carto: {
      type: "raster",
      tiles: ["a", "b", "c", "d"].map((s) => `https://${s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png`),
      tileSize: 256,
      attribution: "© OpenStreetMap contributors © CARTO",
    },
  },
  layers: [{ id: "carto", type: "raster", source: "carto" }],
};

type Props = {
  /** Called once the basemap style has loaded; add sources, layers and handlers here. */
  onLoad?: (map: MapLibreMap) => void;
  height?: number;
  /** Overlays (legends) positioned over the map. */
  children?: ReactNode;
};

/** A MapLibre map on the CARTO dark basemap, centered on Valencia. */
export default function BaseMap({ onLoad, height = 420, children }: Props) {
  const container = useRef<HTMLDivElement>(null);
  const onLoadRef = useRef(onLoad);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    onLoadRef.current = onLoad;
  });

  useEffect(() => {
    if (!container.current) return;
    let map: MapLibreMap;
    try {
      map = new MapLibreMap({
        container: container.current,
        style: BASEMAP,
        center: VALENCIA_CENTER,
        zoom: VALENCIA_ZOOM,
        attributionControl: { compact: true },
      });
    } catch (err) {
      // The constructor throws when WebGL is unavailable.
      console.warn("Map failed to initialise", err);
      // eslint-disable-next-line react-hooks/set-state-in-effect -- the failure is only knowable after mount
      setFailed(true);
      return;
    }
    map.addControl(new NavigationControl({ showCompass: false }), "top-right");
    map.on("load", () => onLoadRef.current?.(map));
    return () => map.remove();
  }, []);

  if (failed) return <EmptyState message={COPY.mapLoadError} />;
  return (
    <div className={styles.wrap} style={{ height }}>
      <div ref={container} className={styles.canvas} />
      {children}
    </div>
  );
}
