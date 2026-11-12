"use client";

import dynamic from "next/dynamic";
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

/** The single switch from a primitive payload to its component. */
export default function Primitive({ p }: { p: PrimitivePayload }) {
  switch (p.kind) {
    case "text":
      return <TextBlock p={p} />;
    case "kpi":
      return <KpiCard p={p} />;
    case "table":
      return <DataTable p={p} />;
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
      console.warn("Unknown primitive kind", (p as { kind?: unknown }).kind);
      return null;
  }
}
