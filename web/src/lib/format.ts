import type { TableColumn } from "./types";

const LOCALE = "es-ES";
const DASH = "—";

const CURRENCY = new Intl.NumberFormat(LOCALE, {
  style: "currency",
  currency: "EUR",
  minimumFractionDigits: 0,
  maximumFractionDigits: 0,
});
const PERCENT = new Intl.NumberFormat(LOCALE, { style: "percent", minimumFractionDigits: 1, maximumFractionDigits: 1 });
const INTEGER = new Intl.NumberFormat(LOCALE, { maximumFractionDigits: 0 });
const NUMBER = new Intl.NumberFormat(LOCALE, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
const COMPACT = new Intl.NumberFormat(LOCALE, { maximumFractionDigits: 2 });

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
