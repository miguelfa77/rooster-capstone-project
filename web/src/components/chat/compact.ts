// Compact density (sidebar and phone sheet): which blocks render inline and how a card names one.
import { parseRows } from "@/components/primitives/DataTable";
import { COPY } from "@/lib/copy";
import { toNumber } from "@/lib/format";
import type { PrimitivePayload } from "@/lib/types";

const TABLE_MAX_COLUMNS = 4;
const TABLE_MAX_ROWS = 6;

/** Text, KPIs and small tables (≤ 4 columns, ≤ 6 rows) fit the column; the rest become cards. */
export function inlineInCompact(p: PrimitivePayload): boolean {
  if (p.kind === "text" || p.kind === "kpi") return true;
  if (p.kind !== "table") return false;
  const rows = parseRows(p.rows_json);
  const columns = p.columns && p.columns.length > 0 ? p.columns.length : Object.keys(rows[0] ?? {}).length;
  return columns <= TABLE_MAX_COLUMNS && rows.length <= TABLE_MAX_ROWS;
}

/** A Plotly figure's own title (`layout.title` as a string or `{ text }`). */
function plotlyTitle(output: string): string | null {
  try {
    const title: unknown = (JSON.parse(output) as { layout?: { title?: unknown } })?.layout?.title;
    if (typeof title === "string") return title;
    if (title && typeof title === "object" && typeof (title as { text?: unknown }).text === "string")
      return (title as { text: string }).text;
  } catch {
    // not a figure
  }
  return null;
}

/** The block's own title/heading/label field, when it has one. */
export function blockTitle(p: PrimitivePayload): string | null {
  let title: string | null = null;
  switch (p.kind) {
    case "bar":
      title = p.value_label;
      break;
    case "scatter":
    case "line":
      title = p.y_label;
      break;
    case "choropleth":
      title = p.metric_label;
      break;
    case "point_map":
      title = p.color_label;
      break;
    case "composite":
      title = p.heading;
      break;
    case "code":
      title = p.output_type === "plotly_json" ? plotlyTitle(p.output) : null;
      break;
  }
  return title?.trim() || null;
}

export function kindLabel(p: PrimitivePayload): string {
  return COPY.blockKind[p.kind];
}

/** What a card draws: bars, one or more lines (y values in order) or scatter points. */
export type Sketch =
  | { kind: "bar"; values: number[] }
  | { kind: "line"; series: number[][] }
  | { kind: "scatter"; points: [number, number][] };

function numbers(v: unknown): number[] | null {
  if (!Array.isArray(v)) return null;
  const out = v.map(toNumber);
  return out.every((n): n is number => n !== null) ? out : null;
}

/** A sketch of a Plotly figure's first trace when it is a simple bar, line or scatter; else null. Never throws. */
export function plotlySketch(output: string): Sketch | null {
  try {
    const trace: unknown = (JSON.parse(output) as { data?: unknown[] } | null)?.data?.[0];
    if (!trace || typeof trace !== "object") return null;
    const t = trace as { type?: unknown; mode?: unknown; orientation?: unknown; x?: unknown; y?: unknown };
    const type = t.type ?? "scatter";
    if (type === "bar") {
      const values = numbers(t.orientation === "h" ? t.x : t.y);
      return values && values.length > 0 ? { kind: "bar", values } : null;
    }
    if (type !== "scatter") return null;
    const y = numbers(t.y);
    if (!y || y.length === 0) return null;
    const mode = typeof t.mode === "string" ? t.mode : "lines";
    if (mode.includes("lines")) return y.length >= 2 ? { kind: "line", series: [y] } : null;
    const x = numbers(t.x);
    if (!x || x.length !== y.length) return null;
    return { kind: "scatter", points: x.map((v, i) => [v, y[i]]) };
  } catch {
    return null;
  }
}
