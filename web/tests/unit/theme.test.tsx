import { act, render, screen } from "@testing-library/react";
import { readFileSync } from "node:fs";
import path from "node:path";
import { useTheme } from "next-themes";
import { beforeAll, beforeEach, describe, expect, it } from "vitest";
import { ThemeProvider } from "@/components/theme-provider";
import * as chartTheme from "@/components/primitives/chartTheme";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";

const CSS = readFileSync(path.resolve(__dirname, "../../src/app/globals.css"), "utf8");

const LEGACY_VARS = [
  "--bg",
  "--card",
  "--border",
  "--text",
  "--muted",
  "--accent",
  "--transit-metro",
  "--transit-bus",
  "--transit-other",
  "--listing-venta",
  "--listing-alquiler",
  "--vut",
];

/** Body of the first top-level rule whose selector is exactly `selector`. */
function block(selector: string): string {
  const escaped = selector.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const match = CSS.match(new RegExp(`(?:^|\\n)${escaped}\\s*\\{([^}]*)\\}`));
  if (!match) throw new Error(`no ${selector} block in globals.css`);
  return match[1];
}

function varValue(body: string, name: string): string | undefined {
  return body.match(new RegExp(`(?:^|[\\s;])${name}\\s*:\\s*([^;]+);`))?.[1].trim();
}

type Mode = ":root" | ".dark";
const MODES: Mode[] = [":root", ".dark"];

/** A token's color in a mode, following var() references (`.dark` falls back to `:root`). */
function color(mode: Mode, name: string): string {
  const raw = varValue(block(mode), name) ?? varValue(block(":root"), name);
  if (!raw) throw new Error(`${name} undefined in ${mode}`);
  const ref = raw.match(/^var\((--[\w-]+)\)$/);
  return ref ? color(mode, ref[1]) : raw.toLowerCase();
}

// WCAG 2.x relative luminance and contrast ratio.
function rgb(hex: string): number[] {
  const m = hex.match(/^#([0-9a-f]{6})$/i);
  if (!m) throw new Error(`not a #rrggbb color: ${hex}`);
  return [0, 2, 4].map((i) => parseInt(m[1].slice(i, i + 2), 16));
}
function luminance(hex: string): number {
  const [r, g, b] = rgb(hex).map((c) => {
    const s = c / 255;
    return s <= 0.03928 ? s / 12.92 : ((s + 0.055) / 1.055) ** 2.4;
  });
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}
function contrast(a: string, b: string): number {
  const [hi, lo] = [luminance(a), luminance(b)].sort((x, y) => y - x);
  return (hi + 0.05) / (lo + 0.05);
}
/** `top` at `alpha` over an opaque `bottom` (e.g. shadcn's `dark:bg-input/30`). */
function over(top: string, alpha: number, bottom: string): string {
  const [t, b] = [rgb(top), rgb(bottom)];
  return "#" + t.map((c, i) => Math.round(c * alpha + b[i] * (1 - alpha)).toString(16).padStart(2, "0")).join("");
}

function Probe() {
  const { theme, setTheme } = useTheme();
  return (
    <>
      <span data-testid="theme">{theme}</span>
      <button onClick={() => setTheme("dark")}>dark</button>
    </>
  );
}

beforeAll(() => {
  // jsdom has no matchMedia; next-themes reads it to resolve "system".
  window.matchMedia = ((query: string) => ({
    matches: false,
    media: query,
    onchange: null,
    addEventListener: () => {},
    removeEventListener: () => {},
    addListener: () => {},
    removeListener: () => {},
    dispatchEvent: () => false,
  })) as typeof window.matchMedia;
});

beforeEach(() => {
  localStorage.clear();
  document.documentElement.className = "";
});

describe("theme", () => {
  it("defaults_to_system_theme", () => {
    render(
      <ThemeProvider>
        <Probe />
      </ThemeProvider>,
    );
    expect(screen.getByTestId("theme")).toHaveTextContent("system");
    expect(document.documentElement).not.toHaveClass("dark");
  });

  it("toggle_sets_dark_class_on_html", () => {
    render(
      <ThemeProvider>
        <Probe />
      </ThemeProvider>,
    );
    act(() => screen.getByRole("button", { name: "dark" }).click());
    expect(document.documentElement).toHaveClass("dark");
    expect(screen.getByTestId("theme")).toHaveTextContent("dark");
  });

  it("legacy_vars_defined_in_both_modes", () => {
    for (const selector of [":root", ".dark"]) {
      const body = block(selector);
      for (const name of LEGACY_VARS) {
        expect(varValue(body, name), `${name} in ${selector}`).toBeTruthy();
      }
    }
  });

  it("primary_is_rooster_red", () => {
    expect(varValue(block(":root"), "--primary")?.toLowerCase()).toBe("#e5484d");
    expect(varValue(block(".dark"), "--primary")?.toLowerCase()).toBe("#e5484d");
  });

  it("tokens_defined_in_both_modes", () => {
    for (const selector of MODES) {
      const body = block(selector);
      for (const name of ["--up", "--down", "--warn", "--ring", "--primary-foreground", "--primary-solid"]) {
        expect(varValue(body, name), `${name} in ${selector}`).toBeTruthy();
      }
    }
  });

  it("num_utility_defined", () => {
    const body = CSS.match(/@utility num\s*\{([^}]*)\}/)?.[1] ?? "";
    expect(body).toContain("var(--font-mono)");
    expect(body).toContain("tabular-nums");
  });

  it("accent_text_contrast_at_least_4_5", () => {
    expect(color(":root", "--accent")).toBe("#cd2b31");
    expect(varValue(block(".dark"), "--accent")).toBe("var(--primary)");
    for (const mode of MODES) {
      for (const bg of ["--background", "--card"]) {
        expect(contrast(color(mode, "--accent"), color(mode, bg)), `${mode} --accent on ${bg}`).toBeGreaterThanOrEqual(4.5);
      }
    }
  });

  it("white_on_primary_solid_at_least_4_5", () => {
    for (const mode of MODES) {
      expect(color(mode, "--primary-solid")).toBe("#cd2b31");
      expect(contrast("#ffffff", color(mode, "--primary-solid"))).toBeGreaterThanOrEqual(4.5);
    }
  });

  it("filled_button_and_badge_use_primary_solid", () => {
    render(
      <>
        <Button>b</Button>
        <Badge>g</Badge>
      </>,
    );
    expect(screen.getByRole("button", { name: "b" }).className).toContain("bg-primary-solid");
    expect(screen.getByText("g").className).toContain("bg-primary-solid");
  });

  it("placeholder_contrast_at_least_4_5", () => {
    for (const mode of MODES) {
      const surfaces = [color(mode, "--background"), color(mode, "--card")];
      // Dark inputs paint --input at 30% over the surface they sit on.
      const inputs = mode === ".dark" ? surfaces.map((s) => over(color(mode, "--input"), 0.3, s)) : [];
      for (const bg of [...surfaces, ...inputs]) {
        expect(contrast(color(mode, "--placeholder"), bg), `${mode} placeholder on ${bg}`).toBeGreaterThanOrEqual(4.5);
      }
    }
  });

  it("chart_series_contrast_at_least_3_on_card", () => {
    for (const series of chartTheme.SERIES_COLORS) {
      const name = series.match(/^var\((--[\w-]+)\)$/)?.[1];
      expect(name, series).toBeTruthy();
      for (const mode of MODES) {
        expect(contrast(color(mode, name!), color(mode, "--card")), `${mode} ${name}`).toBeGreaterThanOrEqual(3);
      }
    }
  });

  it("chart_theme_reads_css_variables", () => {
    const serialized = JSON.stringify(chartTheme);
    expect(serialized).not.toMatch(/#[0-9a-f]{3,8}\b/i);
    expect(chartTheme.ACCENT).toBe("var(--primary)");
    expect(serialized).toContain("var(--");
  });
});
