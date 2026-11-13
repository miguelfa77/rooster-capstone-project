import { describe, expect, it } from "vitest";
import { formatCell, relativeDay, toNumber } from "@/lib/format";

// Same configuration as format.ts: es-ES with grouping forced on (v2 parity, "1.235 €").
const es = (opts: Intl.NumberFormatOptions, v: number) =>
  new Intl.NumberFormat("es-ES", { useGrouping: "always", ...opts }).format(v);

describe("formatCell", () => {
  it("currency_rounds_to_whole_euros", () => {
    const expected = es(
      { style: "currency", currency: "EUR", maximumFractionDigits: 0, minimumFractionDigits: 0 },
      1235,
    );
    expect(formatCell(1234.5, "currency")).toBe(expected);
    expect(expected).toContain("1.235");
    expect(expected.replace(/\s/g, " ")).toBe("1.235 €");
  });

  it("currency_groups_five_digit_numbers", () => {
    expect(formatCell(285000, "currency").replace(/\s/g, " ")).toBe("285.000 €");
  });

  it("percentage_one_decimal_from_percent_value", () => {
    const expected = es({ style: "percent", minimumFractionDigits: 1, maximumFractionDigits: 1 }, 0.06234);
    expect(formatCell(6.234, "percentage")).toBe(expected);
    expect(expected.replace(/\s/g, " ")).toBe("6,2 %");
  });

  it("integer_and_number", () => {
    expect(formatCell(12345.6, "integer")).toBe(es({ maximumFractionDigits: 0 }, 12345.6));
    expect(formatCell(1234, "integer")).toBe("1.234");
    expect(formatCell(1234.5, "number")).toBe("1.234,50");
    expect(formatCell(3.14159, "number")).toBe("3,14");
    expect(formatCell("2.5", "number")).toBe("2,50");
  });

  it("null_and_undefined_render_dash", () => {
    expect(formatCell(null, "currency")).toBe("—");
    expect(formatCell(undefined, "text")).toBe("—");
  });

  it("non_numeric_value_with_numeric_formatter_falls_back_to_string", () => {
    expect(formatCell("n/d", "currency")).toBe("n/d");
  });

  it("link_and_text_return_the_string", () => {
    expect(formatCell("https://www.idealista.com/inmueble/1/", "link")).toBe("https://www.idealista.com/inmueble/1/");
    expect(formatCell("Russafa", "neighborhood-name")).toBe("Russafa");
    expect(formatCell(42, "text")).toBe("42");
  });
});

describe("toNumber", () => {
  it("accepts_finite_numbers_and_numeric_strings_only", () => {
    expect(toNumber(6.1)).toBe(6.1);
    expect(toNumber("6.1")).toBe(6.1);
    expect(toNumber("")).toBeNull();
    expect(toNumber(null)).toBeNull();
    expect(toNumber(Number.NaN)).toBeNull();
    expect(toNumber("abc")).toBeNull();
  });
});

describe("relativeDay", () => {
  const now = new Date(2026, 10, 13, 9, 0);
  const at = (y: number, m: number, d: number, h = 12) => new Date(y, m, d, h).toISOString();

  it("relative_day_in_spanish", () => {
    expect(relativeDay(at(2026, 10, 13, 8), now)).toBe("hoy");
    expect(relativeDay(at(2026, 10, 12, 23), now)).toBe("ayer");
    expect(relativeDay(at(2026, 10, 10), now)).toBe("hace 3 días");
    expect(relativeDay(at(2026, 10, 7), now)).toBe("hace 6 días");
    expect(relativeDay(at(2026, 10, 6), now)).toBe("6 nov");
    expect(relativeDay(at(2025, 11, 24), now)).toBe("24 dic 2025");
  });
});
