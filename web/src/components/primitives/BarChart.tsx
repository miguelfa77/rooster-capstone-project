"use client";

import { Bar, BarChart as RBarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { formatNumber, toNumber } from "@/lib/format";
import type { BarPayload } from "@/lib/types";
import { ACCENT, AXIS, GRID, TOOLTIP } from "./chartTheme";
import EmptyState from "./EmptyState";
import styles from "./Chart.module.css";

export default function BarChart({ p }: { p: BarPayload }) {
  const rows = p.data
    .map((r) => ({ label: r[p.label_field], value: toNumber(r[p.value_field]) }))
    .filter(
      (r): r is { label: unknown; value: number } => r.value !== null && r.label !== null && r.label !== undefined,
    )
    .map((r) => ({ label: String(r.label), value: r.value }))
    .sort((a, b) => b.value - a.value)
    .slice(0, Math.max(0, p.top_n));
  if (rows.length === 0) return <EmptyState />;

  const horizontal = p.orientation === "h";
  const height = horizontal ? Math.max(200, rows.length * 32 + 48) : 320;
  const tooltip = <Tooltip {...TOOLTIP} formatter={(v) => [formatNumber(v), p.value_label]} />;

  return (
    <div className={styles.chart} role="img" aria-label={p.value_label}>
      <ResponsiveContainer width="100%" height={height}>
        {horizontal ? (
          <RBarChart data={rows} layout="vertical" margin={{ left: 8, right: 16 }}>
            <CartesianGrid {...GRID} horizontal={false} />
            <XAxis type="number" {...AXIS} tickFormatter={formatNumber} />
            <YAxis type="category" dataKey="label" width={140} interval={0} {...AXIS} />
            {tooltip}
            <Bar dataKey="value" name={p.value_label} fill={ACCENT} radius={[0, 3, 3, 0]} />
          </RBarChart>
        ) : (
          <RBarChart data={rows} margin={{ right: 16 }}>
            <CartesianGrid {...GRID} vertical={false} />
            <XAxis type="category" dataKey="label" interval={0} {...AXIS} />
            <YAxis type="number" {...AXIS} tickFormatter={formatNumber} />
            {tooltip}
            <Bar dataKey="value" name={p.value_label} fill={ACCENT} radius={[3, 3, 0, 0]} />
          </RBarChart>
        )}
      </ResponsiveContainer>
    </div>
  );
}
