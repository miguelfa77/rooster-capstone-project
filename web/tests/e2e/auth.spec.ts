// Signed-out behaviour. Uses plain Playwright `test` (no session cookie). The Google round trip itself
// cannot run here; see README "Google sign-in".
import { expect, test } from "@playwright/test";

test("signed_out_redirects_to_entrar", async ({ page }) => {
  await page.goto("/chat");
  await expect(page).toHaveURL(/\/entrar\?next=%2Fchat$/);
  await expect(page.getByRole("button", { name: "Continuar con Google" })).toBeVisible();
});

test("entrar_shows_google_button", async ({ page }) => {
  await page.goto("/entrar");
  await expect(page.getByRole("heading", { level: 1, name: "Rooster" })).toBeVisible();
  await expect(page.getByRole("button", { name: "Continuar con Google" })).toBeEnabled();
});

test("invitacion_page_renders", async ({ page }) => {
  await page.goto("/entrar/invitacion");
  await expect(page.getByRole("heading", { name: "Rooster es solo por invitación" })).toBeVisible();
  await expect(page.getByRole("button", { name: "Probar con otra cuenta" })).toBeVisible();
});
