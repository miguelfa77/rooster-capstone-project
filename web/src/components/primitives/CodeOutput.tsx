"use client";

import dynamic from "next/dynamic";
import { useMemo } from "react";
import { COPY } from "@/lib/copy";
import type { CodePayload } from "@/lib/types";
import EmptyState from "./EmptyState";
import type { PlotlyFigureProps } from "./PlotlyFigure";
import styles from "./CodeOutput.module.css";

const PlotlyFigure = dynamic(() => import("./PlotlyFigure"), { ssr: false });

function parseFigure(output: string): PlotlyFigureProps | null {
  try {
    const fig: unknown = JSON.parse(output);
    if (typeof fig === "object" && fig !== null && Array.isArray((fig as PlotlyFigureProps).data))
      return fig as PlotlyFigureProps;
  } catch {
    // fall through
  }
  return null;
}

export default function CodeOutput({ p }: { p: CodePayload }) {
  const figure = useMemo(
    () => (p.output_type === "plotly_json" ? parseFigure(p.output) : null),
    [p.output_type, p.output],
  );

  if (p.output_type === "folium_html") {
    return <iframe className={styles.iframe} sandbox="allow-scripts" srcDoc={p.output} title={COPY.codeMapTitle} />;
  }
  if (!figure) return <EmptyState message={COPY.codeChartError} />;
  return <PlotlyFigure data={figure.data} layout={figure.layout} />;
}
