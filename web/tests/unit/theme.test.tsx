import { act, render, screen } from "@testing-library/react";
import { readFileSync } from "node:fs";
import path from "node:path";
import { useTheme } from "next-themes";
import { beforeAll, beforeEach, describe, expect, it } from "vitest";
import { ThemeProvider } from "@/components/theme-provider";
import * as chartTheme from "@/components/primitives/chartTheme";

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

  it("chart_theme_reads_css_variables", () => {
    const serialized = JSON.stringify(chartTheme);
    expect(serialized).not.toMatch(/#[0-9a-f]{3,8}\b/i);
    expect(chartTheme.ACCENT).toBe("var(--primary)");
    expect(serialized).toContain("var(--");
  });
});
