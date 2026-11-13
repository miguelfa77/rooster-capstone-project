// Compact density (sidebar and phone sheet): which blocks render inline and how a card names one.
import { parseRows } from "@/components/primitives/DataTable";
import { COPY } from "@/lib/copy";
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
    case "kpi":
      title = p.label;
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
