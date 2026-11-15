"use client";

import { Map as MapLibreMap, NavigationControl, type StyleSpecification } from "maplibre-gl";
import { useTheme } from "next-themes";
import { useEffect, useRef, useState, type ReactNode } from "react";
import EmptyState from "@/components/primitives/EmptyState";
import { COPY } from "@/lib/copy";
import styles from "./Map.module.css";

export const VALENCIA_CENTER: [number, number] = [-0.3763, 39.4699];
export const VALENCIA_ZOOM = 12;

type Mode = "light" | "dark";

function basemap(mode: Mode): StyleSpecification {
  return {
    version: 8,
    sources: {
      carto: {
        type: "raster",
        tiles: ["a", "b", "c", "d"].map((s) => `https://${s}.basemaps.cartocdn.com/${mode}_all/{z}/{x}/{y}.png`),
        tileSize: 256,
        attribution: "© OpenStreetMap contributors © CARTO",
      },
    },
    layers: [{ id: "carto", type: "raster", source: "carto" }],
  };
}

/** A theme token's current value; MapLibre paint properties cannot use CSS variables. */
export function tokenColor(name: string, fallback = "gray"): string {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || fallback;
}

type Props = {
  /** Called once the basemap style has loaded; add sources, layers and handlers here. */
  onLoad?: (map: MapLibreMap) => void;
  height?: number;
  /** Overlays (legends) positioned over the map. */
  children?: ReactNode;
};

/**
 * A MapLibre map on the CARTO basemap matching the theme, centered on Valencia. Switching
 * theme recreates the map, so onLoad re-adds overlays with the new theme's token colors.
 */
export default function BaseMap({ onLoad, height = 420, children }: Props) {
  // Undefined until next-themes has resolved the theme; wait, so dark users never get a throwaway light map.
  const { resolvedTheme } = useTheme();
  const mode: Mode | null = resolvedTheme === "dark" ? "dark" : resolvedTheme === "light" ? "light" : null;
  const container = useRef<HTMLDivElement>(null);
  const onLoadRef = useRef(onLoad);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    onLoadRef.current = onLoad;
  });

  useEffect(() => {
    if (!container.current || !mode) return;
    let map: MapLibreMap;
    try {
      map = new MapLibreMap({
        container: container.current,
        style: basemap(mode),
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
  }, [mode]);

  if (failed) return <EmptyState message={COPY.mapLoadError} />;
  return (
    <div className={styles.wrap} style={{ height }}>
      <div ref={container} className={styles.canvas} />
      {children}
    </div>
  );
}
