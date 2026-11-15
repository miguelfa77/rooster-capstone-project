"use client";

import { useTheme } from "next-themes";
import type { Data, Layout } from "plotly.js-dist-min";
import { useEffect, useRef } from "react";
import styles from "./CodeOutput.module.css";

const BASE_LAYOUT: Partial<Layout> = {
  paper_bgcolor: "rgba(0,0,0,0)",
  plot_bgcolor: "rgba(0,0,0,0)",
  margin: { t: 48, r: 16, b: 48, l: 56 },
  autosize: true,
};

/**
 * Plotly cannot read CSS variables, so the theme's text and grid colors are resolved when plotting
 * (under the figure's own axis settings, which win).
 */
function themed(el: HTMLElement, layout: Partial<Layout> | undefined): Partial<Layout> {
  const css = getComputedStyle(el);
  const text = css.getPropertyValue("--foreground").trim();
  const grid = css.getPropertyValue("--border").trim();
  const axis = (own: Partial<Layout["xaxis"]> | undefined) =>
    grid ? { gridcolor: grid, zerolinecolor: grid, linecolor: grid, ...own } : own;
  return {
    ...BASE_LAYOUT,
    ...(text ? { font: { color: text } } : {}),
    ...layout,
    xaxis: axis(layout?.xaxis),
    yaxis: axis(layout?.yaxis),
  };
}

export type PlotlyFigureProps = { data: Data[]; layout?: Partial<Layout> };

/**
 * Renders a Plotly figure; plotly.js itself is imported only when this mounts. The figure is
 * redrawn when the resolved theme changes, and not drawn before it is known.
 */
export default function PlotlyFigure({ data, layout }: PlotlyFigureProps) {
  const ref = useRef<HTMLDivElement>(null);
  const { resolvedTheme } = useTheme();

  useEffect(() => {
    const el = ref.current;
    if (!el || !resolvedTheme) return;
    let cancelled = false;
    let purge: ((root: HTMLElement) => void) | null = null;
    void import("plotly.js-dist-min").then(({ default: Plotly }) => {
      if (cancelled) return;
      purge = Plotly.purge;
      void Plotly.newPlot(el, data, themed(el, layout), { responsive: true, displaylogo: false });
    });
    return () => {
      cancelled = true;
      purge?.(el);
    };
  }, [data, layout, resolvedTheme]);

  return <div ref={ref} className={styles.plotly} />;
}
