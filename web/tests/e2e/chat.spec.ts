import { expect, test } from "@playwright/test";
import { blocks, conversation, expectEveryKind, preparePage, sendMessage, waitForTurnEnd } from "./helpers";

let errors: string[] = [];

test.beforeEach(async ({ page }) => {
  errors = await preparePage(page);
  await page.goto("/");
  await expect(page.getByTestId("chat-empty")).toBeVisible();
});

test.afterEach(() => {
  expect(errors).toEqual([]);
});

test("full_turn_renders_every_kind", async ({ page }) => {
  await sendMessage(page, "todo");
  const conv = conversation(page);
  await expect(conv.getByText("todo", { exact: true })).toBeVisible();
  // The thinking box shows while the turn streams, then prose arrives.
  await expect(page.getByRole("button", { name: /Pensando…/ })).toBeVisible();
  await expect(conv.getByText(/Benimaclet lidera la rentabilidad/)).toBeVisible();

  await waitForTurnEnd(page);
  await expectEveryKind(page);
  await expect(conv.getByText("Rentabilidad media en Valencia").first()).toBeVisible();
  // Stored blocks: 11 top-level primitives, text included (composite children are not wrapped).
  await expect(blocks(page)).toHaveCount(11);
});

test("follow_up_pill_sends", async ({ page }) => {
  await sendMessage(page, "todo");
  await waitForTurnEnd(page);
  await page.getByRole("button", { name: "¿Y en Russafa?" }).click();
  const conv = conversation(page);
  await expect(conv.getByText("¿Y en Russafa?", { exact: true })).toBeVisible();
  await expect(conv.getByText("Respuesta breve sobre: ¿Y en Russafa?.")).toBeVisible();
  await waitForTurnEnd(page);
  // Only the latest message keeps clickable pills; the earlier one shows them as plain text.
  await expect(page.getByRole("button", { name: "¿Y en Russafa?" })).toHaveCount(1);
});

test("reload_restores_identical_history", async ({ page }) => {
  await sendMessage(page, "todo");
  await waitForTurnEnd(page);
  await expectEveryKind(page);
  await sendMessage(page, "hola");
  await waitForTurnEnd(page);
  await expect(conversation(page).getByText("Respuesta breve sobre: hola.")).toBeVisible();
  const before = await blocks(page).count();
  const textBefore = await conversation(page).innerText();

  await page.reload();
  await expect(conversation(page).getByText("Respuesta breve sobre: hola.")).toBeVisible();
  await expectEveryKind(page);
  await expect(blocks(page)).toHaveCount(before);
  expect(before).toBe(12);
  expect(await conversation(page).innerText()).toBe(textBefore);
});

test("error_turn_shows_fallback", async ({ page }) => {
  await sendMessage(page, "error");
  const alert = conversation(page).getByRole("alert");
  await expect(alert).toContainText("No se pudo completar la consulta.");
  await expect(alert).toContainText("Prueba a reformular la pregunta.");
  await expect(page.getByRole("textbox")).toBeEnabled();
});

test("new_conversation_clears", async ({ page }) => {
  await sendMessage(page, "hola");
  await waitForTurnEnd(page);
  await expect(conversation(page).getByText("Respuesta breve sobre: hola.")).toBeVisible();
  await page.getByRole("button", { name: "Nueva conversación" }).click();
  await expect(page.getByTestId("chat-empty")).toBeVisible();
  expect(await page.evaluate(() => localStorage.getItem("rooster.conversationId"))).toBeNull();
  // A reload stays empty: the old conversation is not restored.
  await page.reload();
  await expect(page.getByTestId("chat-empty")).toBeVisible();
});
