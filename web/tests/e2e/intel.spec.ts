import { expect, test } from "@playwright/test";
import { preparePage } from "./helpers";

let errors: string[] = [];

test.beforeEach(async ({ page }) => {
  errors = await preparePage(page);
  await page.goto("/intel");
});

test.afterEach(() => {
  expect(errors).toEqual([]);
});

test("sections_render", async ({ page }) => {
  await expect(page.getByRole("heading", { level: 1 })).toBeVisible();
  // 1 · framing KPIs
  const market = page.getByRole("region", { name: "Mercado · Valencia" });
  await expect(market.getByText("Rentabilidad bruta mediana")).toBeVisible();
  await expect(market.getByText("Barrios >6% rentabilidad")).toBeVisible();
  // 2 · map
  const map = page.getByRole("region", { name: "Mapa de Valencia" });
  await expect(map.locator("canvas.maplibregl-canvas")).toHaveCount(1);
  await expect(map.getByText("No se pudo cargar el mapa.")).toHaveCount(0);
  // 3 · ranking table
  const ranking = page.getByRole("region", { name: /Ranking de oportunidades/ });
  await expect(ranking.locator("tbody tr")).toHaveCount(3);
  await expect(ranking.getByText("Benimaclet")).toBeVisible();
  // 4 · yield vs liquidity scatter
  const scatter = page.getByRole("region", { name: /Rentabilidad frente a liquidez/ });
  await expect(scatter.locator('[role="img"] svg').first()).toBeVisible();
});

test("toggle_each_layer", async ({ page }) => {
  const map = page.getByRole("region", { name: "Mapa de Valencia" });
  await expect(map.locator("canvas.maplibregl-canvas")).toHaveCount(1);
  for (const name of ["Barrios", "Transporte", "VUT", "Anuncios", "Parcelas"]) {
    const box = map.getByRole("checkbox", { name });
    await expect(box).not.toBeChecked();
    await box.check();
    await expect(box).toBeChecked();
    await expect(map.getByText("Cargando…")).toHaveCount(0);
  }
  await expect(map.getByRole("alert")).toHaveCount(0);
  await expect(map.getByRole("combobox")).toBeVisible(); // color-by select appears with barrios on
  await expect(map.getByText(/borde coral = venta/)).toBeVisible();

  // Turning layers off again does not break anything either.
  for (const name of ["Barrios", "Transporte", "VUT", "Anuncios", "Parcelas"]) {
    await map.getByRole("checkbox", { name }).uncheck();
  }
  await expect(map.getByRole("combobox")).toHaveCount(0);
});
