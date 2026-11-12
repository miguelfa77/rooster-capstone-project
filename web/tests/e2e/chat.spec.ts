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

test("leaving_mid_turn_keeps_the_answer", async ({ page }) => {
  // Hold the turn's response until the reader has left the chat, so it finishes while ChatView is unmounted.
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
  await page.getByRole("link", { name: "Inteligencia" }).click();
  await expect(page).toHaveURL(/\/intel$/);
  await expect(conversation(page)).toHaveCount(0);
  release();
  await expect(page.getByRole("region", { name: /Ranking de oportunidades/ }).locator("tbody tr")).toHaveCount(3);

  await page.getByRole("link", { name: "Pregunta a Rooster" }).click();
  await expect(page).toHaveURL(/\/$/);
  await waitForTurnEnd(page);
  await expectEveryKind(page);
  await expect(blocks(page)).toHaveCount(11);
  await expect(page.getByRole("status")).toHaveCount(0);
  expect(await page.evaluate(() => (window as unknown as { __sameDoc?: boolean }).__sameDoc)).toBe(true);
});
