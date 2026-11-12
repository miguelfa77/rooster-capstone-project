import { expect, test } from "@playwright/test";
import { expectEveryKind, pageScrollWidth, preparePage, sendMessage, waitForTurnEnd } from "./helpers";

const WIDTH = 390;

test.use({ viewport: { width: WIDTH, height: 844 }, isMobile: true, hasTouch: true });

let errors: string[] = [];

test.beforeEach(async ({ page }) => {
  errors = await preparePage(page);
});

test.afterEach(() => {
  expect(errors).toEqual([]);
});

test("no_horizontal_page_scroll on / after a full turn", async ({ page }) => {
  await page.goto("/");
  expect(await pageScrollWidth(page)).toBeLessThanOrEqual(WIDTH);
  await sendMessage(page, "todo");
  await waitForTurnEnd(page);
  await expectEveryKind(page);
  expect(await pageScrollWidth(page)).toBeLessThanOrEqual(WIDTH);
  // The composer stays on screen.
  await expect(page.getByRole("textbox")).toBeInViewport();
  await expect(page.getByRole("button", { name: "Enviar" })).toBeInViewport();
});

test("no_horizontal_page_scroll on /intel with layers on", async ({ page }) => {
  await page.goto("/intel");
  const map = page.getByRole("region", { name: "Mapa de Valencia" });
  await expect(page.getByRole("region", { name: /Ranking de oportunidades/ }).locator("tbody tr")).toHaveCount(3);
  for (const name of ["Barrios", "Transporte", "VUT", "Anuncios", "Parcelas"]) {
    await map.getByRole("checkbox", { name }).check();
  }
  await expect(map.getByText("Cargando…")).toHaveCount(0);
  await expect(map.getByRole("combobox")).toBeVisible();
  expect(await pageScrollWidth(page)).toBeLessThanOrEqual(WIDTH);
});
