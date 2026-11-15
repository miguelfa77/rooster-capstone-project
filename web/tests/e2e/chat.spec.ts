import { blocks, conversation, expect, preparePage, sendMessage, test, waitForTurnEnd } from "./helpers";

let errors: string[] = [];

test.beforeEach(async ({ page }) => {
  errors = await preparePage(page);
  await page.goto("/");
  await expect(page.getByRole("complementary", { name: "Chat con Rooster" }).getByTestId("chat-empty")).toBeVisible();
});

test.afterEach(() => {
  expect(errors).toEqual([]);
});

test("sidebar_turn_renders_compact_then_full_dialog", async ({ page }) => {
  await sendMessage(page, "todo");
  const conv = conversation(page);
  await expect(conv.getByText("todo", { exact: true })).toBeVisible();
  await expect(conv.getByText(/Benimaclet lidera la rentabilidad/)).toBeVisible();
  await waitForTurnEnd(page);
  // Stored blocks: 11 top-level primitives, text included.
  await expect(blocks(page)).toHaveCount(11);
  // Big blocks are compact cards in the sidebar: nothing heavy is rendered until one is opened.
  await expect(conv.locator("canvas.maplibregl-canvas")).toHaveCount(0);
  await expect(conv.locator(".js-plotly-plot")).toHaveCount(0);
  await expect(conv.getByRole("button", { name: "¿Y en Russafa?" })).toBeVisible();

  await conv.getByRole("button", { name: /Mapa de barrios/ }).click();
  const dialog = page.getByRole("dialog");
  await expect(dialog.locator("canvas.maplibregl-canvas")).toBeVisible();
  await expect(dialog.getByText("No se pudo cargar el mapa.")).toHaveCount(0);
  await page.keyboard.press("Escape");
  await expect(dialog).toHaveCount(0);
});

test("expand_to_full_screen_keeps_conversation", async ({ page }) => {
  await sendMessage(page, "todo");
  await waitForTurnEnd(page);
  await page.getByRole("button", { name: "Abrir a pantalla completa" }).click();
  await expect(page).toHaveURL(/\/chat\/[0-9a-f-]{36}$/);
  const conv = conversation(page);
  await expect(conv.getByText("todo", { exact: true })).toBeVisible();
  await expect(blocks(page)).toHaveCount(11);
  // Full density renders the blocks themselves.
  await expect(conv.locator("table").first()).toBeVisible();
  await expect(conv.locator("canvas.maplibregl-canvas")).toHaveCount(2);
  await expect(conv.locator(".js-plotly-plot")).toHaveCount(1);
  await expect(conv.locator('iframe[sandbox="allow-scripts"]')).toHaveCount(1);
  await expect(conv.getByRole("heading", { level: 3, name: "Informe: Benimaclet" })).toBeVisible();

  // The same conversation after a reload, and back to the page the user came from.
  await page.reload();
  await expect(conversation(page).getByText("todo", { exact: true })).toBeVisible();
  await expect(blocks(page)).toHaveCount(11);
  await page.getByRole("button", { name: "Volver" }).click();
  await expect(page).toHaveURL(/\/$/);
  await expect(page.getByRole("complementary", { name: "Chat con Rooster" })).toBeVisible();
  await expect(conversation(page).getByText("todo", { exact: true })).toBeVisible();
});

test("error_turn_shows_fallback", async ({ page }) => {
  await sendMessage(page, "error");
  const alert = conversation(page).getByRole("alert");
  await expect(alert).toContainText("No se pudo completar la consulta.");
  await expect(alert).toContainText("Prueba a reformular la pregunta.");
  await expect(page.getByRole("textbox")).toBeEnabled();
});

test("leaving_mid_turn_keeps_the_answer", async ({ page }) => {
  // Hold the turn's response until the reader has navigated away, so it finishes after the page changed.
  let release!: () => void;
  const released = new Promise<void>((r) => (release = r));
  await page.route(/\/api\/rooster\/conversations\/[^/]+\/messages$/, async (route) => {
    const response = await route.fetch();
    const body = await response.body();
    await released;
    await route.fulfill({ response, body });
  });
  // Marks this document; a full reload would drop it.
  await page.evaluate(() => ((window as unknown as { __sameDoc: boolean }).__sameDoc = true));

  await sendMessage(page, "todo");
  await expect(page.getByRole("button", { name: /Pensando…/ })).toBeVisible();
  await page.getByRole("link", { name: "Ver todas" }).first().click();
  await expect(page).toHaveURL(/\/chat$/);
  release();
  await waitForTurnEnd(page);
  await expect(blocks(page)).toHaveCount(11);
  await expect(page.getByRole("status")).toHaveCount(0);
  expect(await page.evaluate(() => (window as unknown as { __sameDoc?: boolean }).__sameDoc)).toBe(true);
});

test("theme_toggle_persists_after_reload", async ({ page }) => {
  const html = page.locator("html");
  await page.getByRole("button", { name: "Menú de usuario" }).click();
  // The preference is saved in the background; wait until the API has it before reloading.
  const saved = page.waitForResponse((r) => r.url().endsWith("/preferences") && r.request().method() === "PATCH");
  await page.getByRole("menuitemradio", { name: "Oscuro" }).click();
  await expect(html).toHaveClass(/dark/);
  expect((await saved).ok()).toBe(true);
  await page.reload();
  await expect(html).toHaveClass(/dark/);
  await page.getByRole("button", { name: "Menú de usuario" }).click();
  await expect(page.getByRole("menuitemradio", { name: "Oscuro" })).toBeChecked();
});
