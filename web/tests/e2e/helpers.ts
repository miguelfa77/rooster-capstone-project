import { expect, test as base, type Browser, type BrowserContext, type Page } from "@playwright/test";
import { encode } from "next-auth/jwt";

const BASE_URL = `http://127.0.0.1:${process.env.E2E_WEB_PORT ?? 3100}`; // same ports as playwright.config.ts
const FAKE_API = `http://localhost:${process.env.E2E_API_PORT ?? 8787}`;
const AUTH_SECRET = "test-secret"; // same as playwright.config.ts
const SESSION_COOKIE = "authjs.session-token"; // Auth.js on http; the cookie name is also its JWT salt

export type TestUser = { uid: string; email: string; name: string };
// uid / email match the profiles in tests/fake-api/server.mjs; both emails are on ROOSTER_ALLOWED_EMAILS.
export const ANA: TestUser = { uid: "u-ana", email: "ana@fund.com", name: "Ana Prueba" };
export const LUIS: TestUser = { uid: "u-luis", email: "luis@fund.com", name: "Luis Prueba" };

/** Adds a valid Auth.js session cookie for `user`, minted with the test secret (no test mode in the app). */
export async function signInAs(context: BrowserContext, user: TestUser) {
  const value = await encode({
    token: { sub: user.uid, uid: user.uid, email: user.email, name: user.name },
    secret: AUTH_SECRET,
    salt: SESSION_COOKIE,
  });
  await context.addCookies([{ name: SESSION_COOKIE, value, url: BASE_URL }]);
}

/** A second browser context, signed in as `user`, with the same stubs as `preparePage`. */
export async function newSignedInPage(browser: Browser, user: TestUser) {
  const context = await browser.newContext({ baseURL: BASE_URL });
  await signInAs(context, user);
  const page = await context.newPage();
  const errors = await preparePage(page);
  return { context, page, errors };
}

/** Signed in as Ana by default; the fake API starts every spec with no conversations or preferences. */
export const test = base.extend({
  context: async ({ context }, use) => {
    await fetch(`${FAKE_API}/__reset`, { method: "POST" });
    await signInAs(context, ANA);
    await use(context);
  },
});
export { expect };


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

/** One wrapper per rendered top-level block (AssistantMessage's `message-block`). */
export const blocks = (page: Page) => conversation(page).getByTestId("message-block");

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
