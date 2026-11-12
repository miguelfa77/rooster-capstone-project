import { expect, type Page } from "@playwright/test";

// 1×1 transparent PNG served for basemap tiles, so tests never depend on the CARTO CDN.
const EMPTY_TILE = Buffer.from(
  "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=",
  "base64",
);

/** Stubs basemap tiles and collects uncaught page errors; call before the first navigation. */
export async function preparePage(page: Page): Promise<string[]> {
  const errors: string[] = [];
  page.on("pageerror", (err) => errors.push(err.message));
  await page.route(/basemaps\.cartocdn\.com/, (route) =>
    route.fulfill({ status: 200, contentType: "image/png", body: EMPTY_TILE }),
  );
  return errors;
}

export const conversation = (page: Page) => page.getByRole("region", { name: "Conversación" });

/** One wrapper per rendered primitive (the `block` class of AssistantMessage). */
export const blocks = (page: Page) => conversation(page).locator('[class*="_block"]');

export async function sendMessage(page: Page, text: string) {
  const input = page.getByRole("textbox", { name: "Pregunta por anuncios, rentabilidades, barrios…" });
  await expect(input).toBeEnabled();
  await input.fill(text);
  await page.getByRole("button", { name: "Enviar" }).click();
}

/** Waits until the turn is over: the composer is enabled again and the live thinking box is gone. */
export async function waitForTurnEnd(page: Page) {
  await expect(page.getByRole("textbox")).toBeEnabled();
  await expect(page.getByRole("button", { name: /Pensando…/ })).toHaveCount(0);
}

/** Every primitive of the "todo" turn has rendered. */
export async function expectEveryKind(page: Page) {
  const conv = conversation(page);
  await expect(conv.locator("table").first()).toBeVisible();
  await expect(conv.locator('[role="img"] svg').first()).toBeVisible();
  await expect(conv.locator("canvas.maplibregl-canvas")).toHaveCount(2);
  await expect(conv.getByText("No se pudo cargar el mapa.")).toHaveCount(0);
  await expect(conv.locator(".js-plotly-plot")).toHaveCount(1);
  await expect(conv.locator('iframe[sandbox="allow-scripts"]')).toHaveCount(1);
  await expect(conv.getByRole("heading", { level: 3, name: "Informe: Benimaclet" })).toBeVisible();
}

/** Page-level horizontal overflow; inner scroll containers do not count. */
export const pageScrollWidth = (page: Page) => page.evaluate(() => document.documentElement.scrollWidth);
