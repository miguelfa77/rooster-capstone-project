"use client";

import dynamic from "next/dynamic";
import { useEffect } from "react";
import type { PrimitivePayload } from "@/lib/types";
import BarChart from "./BarChart";
import CodeOutput from "./CodeOutput";
import Composite from "./Composite";
import DataTable from "./DataTable";
import KpiCard from "./KpiCard";
import LineChart from "./LineChart";
import ScatterChart from "./ScatterChart";
import TextBlock from "./TextBlock";

const ChoroplethMap = dynamic(() => import("@/components/map/ChoroplethMap"), { ssr: false });
const PointMap = dynamic(() => import("@/components/map/PointMap"), { ssr: false });

const KNOWN_KINDS: ReadonlySet<string> = new Set<PrimitivePayload["kind"]>([
  "text",
  "kpi",
  "table",
  "code",
  "bar",
  "scatter",
  "line",
  "choropleth",
  "point_map",
  "composite",
]);

/** The single switch from a primitive payload to its component. `compact` tightens tables for the narrow chat column. */
export default function Primitive({ p, compact = false }: { p: PrimitivePayload; compact?: boolean }) {
  const kind: unknown = p.kind;
  useEffect(() => {
    if (typeof kind !== "string" || !KNOWN_KINDS.has(kind)) console.warn("Unknown primitive kind", kind);
  }, [kind]);

  switch (p.kind) {
    case "text":
      return <TextBlock p={p} />;
    case "kpi":
      return <KpiCard p={p} />;
    case "table":
      return <DataTable p={p} compact={compact} />;
    case "code":
      return <CodeOutput p={p} />;
    case "bar":
      return <BarChart p={p} />;
    case "scatter":
      return <ScatterChart p={p} />;
    case "line":
      return <LineChart p={p} />;
    case "choropleth":
      return <ChoroplethMap p={p} />;
    case "point_map":
      return <PointMap p={p} />;
    case "composite":
      return <Composite p={p} />;
    default:
      return null;
  }
}
