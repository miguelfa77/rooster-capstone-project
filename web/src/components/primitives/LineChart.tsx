"use client";

import {
  CartesianGrid,
  Legend,
  Line,
  LineChart as RLineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { formatNumber, toNumber } from "@/lib/format";
import type { LinePayload } from "@/lib/types";
import { AXIS, GRID, LEGEND, TOOLTIP, seriesColor } from "./chartTheme";
import EmptyState from "./EmptyState";
import styles from "./Chart.module.css";

const X = "__x";

export default function LineChart({ p }: { p: LinePayload }) {
  const rows = p.data.filter(
    (r) => r[p.x_field] !== null && r[p.x_field] !== undefined && toNumber(r[p.y_field]) !== null,
  );
  if (rows.length === 0) return <EmptyState />;

  // Pivot to one record per x with one key per series, which is what Recharts expects.
  // Walk rows in their original order so the x axis keeps it even when a series has gaps.
  const colorField = p.color_field;
  const seriesOf = (r: (typeof rows)[number]) => (colorField ? String(r[colorField] ?? "—") : p.y_label);
  const series: string[] = [];
  const byX = new Map<string, Record<string, unknown>>();
  for (const r of rows) {
    const name = seriesOf(r);
    if (!series.includes(name)) series.push(name);
    const x = String(r[p.x_field]);
    const rec = byX.get(x) ?? { [X]: x };
    rec[name] = toNumber(r[p.y_field]);
    byX.set(x, rec);
  }
  const data = [...byX.values()];

  return (
    <div className={styles.chart} role="img" aria-label={p.y_label}>
      <ResponsiveContainer width="100%" height={300}>
        <RLineChart data={data} margin={{ right: 16 }}>
          <CartesianGrid {...GRID} vertical={false} />
          <XAxis dataKey={X} {...AXIS} />
          <YAxis {...AXIS} tickFormatter={formatNumber} domain={["auto", "auto"]} />
          <Tooltip {...TOOLTIP} cursor={{ stroke: AXIS.stroke }} formatter={(v) => formatNumber(v)} />
          {series.length > 1 ? <Legend {...LEGEND} verticalAlign="top" /> : null}
          {series.map((name, i) => (
            <Line
              key={name}
              type="monotone"
              dataKey={name}
              name={name}
              stroke={seriesColor(i)}
              strokeWidth={2}
              dot={{ r: 3 }}
              connectNulls
              isAnimationActive={false}
            />
          ))}
        </RLineChart>
      </ResponsiveContainer>
    </div>
  );
}
