import type { TableColumn } from "./types";

const LOCALE = "es-ES";
// es-ES leaves 4-digit numbers ungrouped by default; force grouping for parity with v2 ("1.235 €").
const GROUPING = { useGrouping: "always" } as const;
const DASH = "—";

const CURRENCY = new Intl.NumberFormat(LOCALE, {
  ...GROUPING,
  style: "currency",
  currency: "EUR",
  minimumFractionDigits: 0,
  maximumFractionDigits: 0,
});
const PERCENT = new Intl.NumberFormat(LOCALE, { style: "percent", minimumFractionDigits: 1, maximumFractionDigits: 1 });
const INTEGER = new Intl.NumberFormat(LOCALE, { ...GROUPING, maximumFractionDigits: 0 });
const NUMBER = new Intl.NumberFormat(LOCALE, { ...GROUPING, minimumFractionDigits: 2, maximumFractionDigits: 2 });
const COMPACT = new Intl.NumberFormat(LOCALE, { ...GROUPING, maximumFractionDigits: 2 });

/** Finite number, or numeric string coerced to one; anything else is null. */
export function toNumber(value: unknown): number | null {
  if (typeof value === "number") return Number.isFinite(value) ? value : null;
  if (typeof value === "string" && value.trim() !== "") {
    const n = Number(value);
    return Number.isFinite(n) ? n : null;
  }
  return null;
}

function asText(value: unknown): string {
  return typeof value === "object" ? JSON.stringify(value) : String(value);
}

/** Format one table cell. Percentages arrive as percent values (6.2 means 6.2 %). */
export function formatCell(value: unknown, formatter: TableColumn["formatter"]): string {
  if (value === null || value === undefined) return DASH;
  const numeric =
    formatter === "currency" || formatter === "percentage" || formatter === "integer" || formatter === "number";
  if (!numeric) return asText(value);
  const n = toNumber(value);
  if (n === null) return asText(value);
  switch (formatter) {
    case "currency":
      return CURRENCY.format(n);
    case "percentage":
      return PERCENT.format(n / 100);
    case "integer":
      return INTEGER.format(n);
    default:
      return NUMBER.format(n);
  }
}

/** es-ES number with up to two decimals, for chart axes and tooltips. */
export function formatNumber(value: unknown): string {
  if (value === null || value === undefined) return DASH;
  const n = toNumber(value);
  return n === null ? asText(value) : COMPACT.format(n);
}

const DAY_MS = 86_400_000;
const RELATIVE = new Intl.RelativeTimeFormat("es", { numeric: "auto" });

/** "hoy", "ayer", "hace N días" within a week (by calendar day); older ⇒ "6 nov" ("24 dic 2025" in another year). */
export function relativeDay(iso: string, now: Date = new Date()): string {
  const date = new Date(iso);
  const startOf = (d: Date) => new Date(d.getFullYear(), d.getMonth(), d.getDate()).getTime();
  const days = Math.round((startOf(now) - startOf(date)) / DAY_MS);
  if (days >= 0 && days < 7) return RELATIVE.format(-days, "day");
  return date.toLocaleDateString("es", {
    day: "numeric",
    month: "short",
    ...(date.getFullYear() === now.getFullYear() ? {} : { year: "numeric" }),
  });
}
