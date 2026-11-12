"use client";

import {
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Scatter,
  ScatterChart as RScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  type TooltipContentProps,
} from "recharts";
import { formatNumber, toNumber } from "@/lib/format";
import type { ScatterPayload } from "@/lib/types";
import { AXIS, GRID, LEGEND, groupBy, seriesColor } from "./chartTheme";
import EmptyState from "./EmptyState";
import styles from "./Chart.module.css";

type Point = { x: number; y: number; row: Record<string, unknown> };

export default function ScatterChart({ p }: { p: ScatterPayload }) {
  const points = p.data
    .map((row) => ({ x: toNumber(row[p.x_field]), y: toNumber(row[p.y_field]), row }))
    .filter((pt): pt is Point => pt.x !== null && pt.y !== null);
  if (points.length === 0) return <EmptyState />;

  const colorField = p.color_field;
  const series: [string, Point[]][] = colorField ? groupBy(points, (pt) => pt.row[colorField]) : [[p.y_label, points]];

  const renderTooltip = ({ active, payload }: TooltipContentProps) => {
    const pt = active ? (payload?.[0]?.payload as Point | undefined) : undefined;
    if (!pt) return null;
    const label = p.label_field ? pt.row[p.label_field] : null;
    return (
      <div className={styles.tooltip}>
        {label !== null && label !== undefined ? <div className={styles.tooltipTitle}>{String(label)}</div> : null}
        <div>
          {p.x_label}: {formatNumber(pt.x)}
        </div>
        <div>
          {p.y_label}: {formatNumber(pt.y)}
        </div>
      </div>
    );
  };

  return (
    <div className={styles.chart} role="img" aria-label={`${p.y_label} / ${p.x_label}`}>
      <ResponsiveContainer width="100%" height={320}>
        <RScatterChart margin={{ right: 16, bottom: 16 }}>
          <CartesianGrid {...GRID} />
          <XAxis
            type="number"
            dataKey="x"
            name={p.x_label}
            {...AXIS}
            tickFormatter={formatNumber}
            label={{ value: p.x_label, position: "insideBottom", offset: -8, fill: AXIS.stroke, fontSize: 12 }}
          />
          <YAxis
            type="number"
            dataKey="y"
            name={p.y_label}
            {...AXIS}
            tickFormatter={formatNumber}
            domain={["auto", "auto"]}
          />
          <Tooltip cursor={{ strokeDasharray: "3 3" }} content={renderTooltip} />
          {series.length > 1 ? <Legend {...LEGEND} verticalAlign="top" /> : null}
          {series.map(([name, pts], i) => (
            <Scatter key={name} name={name} data={pts} fill={seriesColor(i)} />
          ))}
        </RScatterChart>
      </ResponsiveContainer>
    </div>
  );
}
