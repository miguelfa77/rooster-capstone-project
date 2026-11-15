// Parked for the Phase 2 Mapa page; not mounted anywhere.
"use client";

import {
  CartesianGrid,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
  type TooltipContentProps,
} from "recharts";
import { ACCENT, AXIS, GRID } from "@/components/primitives/chartTheme";
import chartStyles from "@/components/primitives/Chart.module.css";
import EmptyState from "@/components/primitives/EmptyState";
import { COPY } from "@/lib/copy";
import { formatCell, formatNumber, toNumber } from "@/lib/format";
import type { YieldLiquidityRow } from "@/lib/types";
import styles from "./Intel.module.css";
import Section from "./Section";
import { useIntel } from "./useIntel";

type Point = { x: number; y: number; z: number | null; name: string };

function toPoints(rows: YieldLiquidityRow[]): Point[] {
  const out: Point[] = [];
  for (const r of rows) {
    const x = toNumber(r.listing_count);
    const y = toNumber(r.yield_pct);
    if (x === null || y === null) continue;
    out.push({ x, y, z: toNumber(r.eur_per_sqm), name: r.neighborhood_name ?? "—" });
  }
  return out;
}

function renderTooltip({ active, payload }: TooltipContentProps) {
  const pt = active ? (payload?.[0]?.payload as Point | undefined) : undefined;
  if (!pt) return null;
  return (
    <div className={chartStyles.tooltip}>
      <div className={chartStyles.tooltipTitle}>{pt.name}</div>
      <div>
        {COPY.intelScatterX}: {formatNumber(pt.x)}
      </div>
      <div>
        {COPY.intelScatterY}: {formatNumber(pt.y)}
      </div>
      <div>€/m²: {formatCell(pt.z, "currency")}</div>
    </div>
  );
}

function Chart({ points }: { points: Point[] }) {
  return (
    <>
      <div className={chartStyles.chart} role="img" aria-label={`${COPY.intelScatterY} / ${COPY.intelScatterX}`}>
        <ResponsiveContainer width="100%" height={360}>
          <ScatterChart margin={{ right: 16, bottom: 16 }}>
            <CartesianGrid {...GRID} />
            <XAxis
              type="number"
              dataKey="x"
              {...AXIS}
              tickFormatter={formatNumber}
              label={{ value: COPY.intelScatterX, position: "insideBottom", offset: -8, fill: AXIS.stroke, fontSize: 12 }}
            />
            <YAxis type="number" dataKey="y" {...AXIS} tickFormatter={formatNumber} domain={["auto", "auto"]} />
            {/* Point size follows median €/m² of sales, as the caption says. */}
            <ZAxis type="number" dataKey="z" range={[40, 400]} />
            <Tooltip cursor={{ strokeDasharray: "3 3" }} content={renderTooltip} />
            <Scatter data={points} fill={ACCENT} fillOpacity={0.85} />
          </ScatterChart>
        </ResponsiveContainer>
      </div>
      <p className={styles.caption}>{COPY.intelScatterCaption}</p>
    </>
  );
}

/** Section 4: yield (y) against liquidity = total listings (x), per barrio. */
export default function YieldLiquidity() {
  const s = useIntel<YieldLiquidityRow[]>("yield-liquidity");
  let body;
  if (s.status === "loading") body = <p className={styles.status}>{COPY.intelLoading}</p>;
  else if (s.status === "error") body = <EmptyState message={COPY.intelLoadError} />;
  else {
    const points = toPoints(s.data);
    // As in Streamlit: fewer than two barrios with a yield is not a chart.
    body = points.length < 2 ? <EmptyState message={COPY.intelNoScatter} /> : <Chart points={points} />;
  }
  return <Section title={COPY.intelS4Title}>{body}</Section>;
}
