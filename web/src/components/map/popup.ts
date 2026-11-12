// Popup bodies built as DOM nodes so row values are always text, never markup.
import { formatCell, toNumber } from "@/lib/format";
import { COPY } from "@/lib/copy";
import type { TooltipField } from "@/lib/types";

function formatValue(value: unknown): string {
  const n = typeof value === "number" ? toNumber(value) : null;
  if (n === null) return formatCell(value, "text");
  return formatCell(n, Number.isInteger(n) ? "integer" : "number");
}

function safeHttpUrl(value: unknown): string | null {
  if (typeof value !== "string") return null;
  try {
    const url = new URL(value);
    return url.protocol === "https:" || url.protocol === "http:" ? url.href : null;
  } catch {
    return null;
  }
}

/** One "label: value" line per field; with `linkField`, a safe external link instead of that field's raw text. */
export function popupContent(
  fields: TooltipField[],
  properties: Record<string, unknown>,
  linkField?: string,
): HTMLElement {
  const root = document.createElement("div");
  for (const { field, label } of fields) {
    if (field === linkField) continue;
    const line = document.createElement("div");
    line.textContent = `${label}: ${formatValue(properties[field])}`;
    root.appendChild(line);
  }
  const href = linkField ? safeHttpUrl(properties[linkField]) : null;
  if (href) {
    const a = document.createElement("a");
    a.href = href;
    a.target = "_blank";
    a.rel = "noopener noreferrer";
    a.textContent = COPY.popupIdealista;
    root.appendChild(a);
  }
  return root;
}
