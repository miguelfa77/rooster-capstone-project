"use client";

import type { Data, Layout } from "plotly.js-dist-min";
import { useEffect, useRef } from "react";
import styles from "./CodeOutput.module.css";

const DARK_LAYOUT: Partial<Layout> = {
  paper_bgcolor: "rgba(0,0,0,0)",
  plot_bgcolor: "rgba(0,0,0,0)",
  font: { color: "#f0f0f0" },
  margin: { t: 48, r: 16, b: 48, l: 56 },
  autosize: true,
};

export type PlotlyFigureProps = { data: Data[]; layout?: Partial<Layout> };

/** Renders a Plotly figure; plotly.js itself is imported only when this mounts. */
export default function PlotlyFigure({ data, layout }: PlotlyFigureProps) {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    let cancelled = false;
    let purge: ((root: HTMLElement) => void) | null = null;
    void import("plotly.js-dist-min").then(({ default: Plotly }) => {
      if (cancelled) return;
      purge = Plotly.purge;
      void Plotly.newPlot(el, data, { ...DARK_LAYOUT, ...layout }, { responsive: true, displaylogo: false });
    });
    return () => {
      cancelled = true;
      purge?.(el);
    };
  }, [data, layout]);

  return <div ref={ref} className={styles.plotly} />;
}
