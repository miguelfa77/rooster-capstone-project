// Recharts styling shared by Bar/Scatter/Line. Colors are CSS variables from globals.css,
// resolved by the browser at render time, so charts follow the light/dark theme.
export const ACCENT = "var(--primary)";
export const SERIES_COLORS = [
  ACCENT,
  "var(--listing-alquiler)",
  "var(--vut)",
  "var(--transit-bus)",
  "var(--transit-metro)",
  "var(--muted-foreground)",
];
export const AXIS = { stroke: "var(--muted-foreground)", tick: { fill: "var(--muted-foreground)", fontSize: 12 } };
export const GRID = { stroke: "var(--border)", strokeDasharray: "3 3" };
export const TOOLTIP = {
  contentStyle: {
    background: "var(--popover)",
    border: "1px solid var(--border)",
    borderRadius: 6,
    color: "var(--popover-foreground)",
  },
  labelStyle: { color: "var(--popover-foreground)" },
  itemStyle: { color: "var(--popover-foreground)" },
  cursor: { fill: "var(--muted-surface)", fillOpacity: 0.5 },
};
export const LEGEND = { wrapperStyle: { color: "var(--muted-foreground)", fontSize: 12 } };

export function seriesColor(i: number): string {
  return SERIES_COLORS[i % SERIES_COLORS.length];
}

/** Group items by a stringified key, preserving first-appearance order. */
export function groupBy<T>(items: T[], keyOf: (item: T) => unknown): [string, T[]][] {
  const groups = new Map<string, T[]>();
  for (const row of items) {
    const key = String(keyOf(row) ?? "—");
    const list = groups.get(key);
    if (list) list.push(row);
    else groups.set(key, [row]);
  }
  return [...groups];
}
