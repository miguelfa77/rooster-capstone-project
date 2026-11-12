// Recharts styling shared by Bar/Scatter/Line (mirrors the palette in globals.css).
export const ACCENT = "#e03030";
export const SERIES_COLORS = [ACCENT, "#48a9da", "#ef9f27", "#1d9e75", "#534ab7", "#a8a8a8"];
export const AXIS = { stroke: "#a8a8a8", tick: { fill: "#a8a8a8", fontSize: 12 } };
export const GRID = { stroke: "#2a2a2a", strokeDasharray: "3 3" };
export const TOOLTIP = {
  contentStyle: { background: "#161616", border: "1px solid #2a2a2a", borderRadius: 6, color: "#f0f0f0" },
  labelStyle: { color: "#f0f0f0" },
  itemStyle: { color: "#f0f0f0" },
  cursor: { fill: "rgba(255,255,255,0.04)" },
};
export const LEGEND = { wrapperStyle: { color: "#a8a8a8", fontSize: 12 } };

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
