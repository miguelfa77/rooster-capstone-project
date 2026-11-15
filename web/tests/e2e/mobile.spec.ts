import { conversation, expect, pageScrollWidth, preparePage, sendMessage, test, waitForTurnEnd } from "./helpers";

const WIDTH = 390;

test.use({ viewport: { width: WIDTH, height: 844 }, isMobile: true, hasTouch: true });

let errors: string[] = [];

test.beforeEach(async ({ page }) => {
  errors = await preparePage(page);
});

test.afterEach(() => {
  expect(errors).toEqual([]);
});

test("no_horizontal_scroll_on_inicio_and_chat", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByRole("button", { name: "Abrir chat" })).toBeVisible();
  expect(await pageScrollWidth(page)).toBeLessThanOrEqual(WIDTH);

  const sheet = page.getByRole("dialog", { name: "Chat con Rooster" });
  await page.getByRole("button", { name: "Abrir chat" }).click();
  await sendMessage(page, "todo");
  await waitForTurnEnd(page);
  await expect(sheet.getByRole("button", { name: "¿Y en Russafa?" })).toBeVisible();
  expect(await pageScrollWidth(page)).toBeLessThanOrEqual(WIDTH);

  // Full-screen chat with every block rendered.
  await sheet.getByRole("button", { name: "Abrir a pantalla completa" }).click();
  await expect(page).toHaveURL(/\/chat\/[0-9a-f-]{36}$/);
  await expect(conversation(page).locator("canvas.maplibregl-canvas")).toHaveCount(2);
  await expect(conversation(page).locator(".js-plotly-plot")).toHaveCount(1);
  expect(await pageScrollWidth(page)).toBeLessThanOrEqual(WIDTH);
  await expect(page.getByRole("textbox")).toBeInViewport();
});

test("chat_opens_as_sheet_and_sends", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByRole("complementary", { name: "Chat con Rooster" })).toBeHidden();
  await page.getByRole("button", { name: "Abrir chat" }).click();
  const sheet = page.getByRole("dialog", { name: "Chat con Rooster" });
  await expect(sheet).toBeVisible();
  await sendMessage(page, "hola");
  await expect(sheet.getByText("Respuesta breve sobre: hola.")).toBeVisible();
  await waitForTurnEnd(page);
  await sheet.getByRole("button", { name: "Cerrar chat" }).click();
  await expect(sheet).toHaveCount(0);
});
