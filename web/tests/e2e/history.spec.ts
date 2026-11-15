import { ANA, LUIS, blocks, conversation, expect, newSignedInPage, preparePage, sendMessage, test, waitForTurnEnd } from "./helpers";

let errors: string[] = [];

test.beforeEach(async ({ page }) => {
  errors = await preparePage(page);
  await page.goto("/");
  await expect(page.getByRole("complementary", { name: "Chat con Rooster" }).getByTestId("chat-empty")).toBeVisible();
});

test.afterEach(() => {
  expect(errors).toEqual([]);
});

test("new_conversation_appears_in_nav_with_title", async ({ page }) => {
  const nav = page.getByRole("navigation", { name: "Navegación principal" }).locator("xpath=..");
  await expect(nav.getByText("Sin conversaciones todavía")).toBeVisible();
  await sendMessage(page, "hola");
  await waitForTurnEnd(page);
  // The title is the first message.
  await expect(nav.getByRole("button", { name: "hola", exact: true })).toBeVisible();
  await expect(nav.getByText("Sin conversaciones todavía")).toHaveCount(0);

  // A second one is listed too, and the first is still there.
  await page.getByRole("button", { name: "Nueva conversación" }).first().click();
  await sendMessage(page, "adiós");
  await waitForTurnEnd(page);
  await expect(nav.getByRole("button", { name: "adiós", exact: true })).toBeVisible();
  await expect(nav.getByRole("button", { name: "hola", exact: true })).toBeVisible();
});

test("rename_and_delete_from_chat_page", async ({ page }) => {
  await sendMessage(page, "hola");
  await waitForTurnEnd(page);
  await page.getByRole("link", { name: "Ver todas" }).first().click();
  await expect(page).toHaveURL(/\/chat$/);

  const list = page.getByRole("main");
  await list.getByRole("button", { name: "Renombrar «hola»" }).click();
  const input = page.getByRole("textbox", { name: "Título de la conversación" });
  await input.fill("Mi análisis de Russafa");
  await input.press("Enter");
  await expect(list.getByRole("button", { name: "Mi análisis de Russafa", exact: true })).toBeVisible();

  await page.reload();
  const row = list.getByRole("button", { name: "Mi análisis de Russafa", exact: true });
  await expect(row).toBeVisible();
  // Opening it goes to the full-screen chat with its messages.
  await row.click();
  await expect(page).toHaveURL(/\/chat\/[0-9a-f-]{36}$/);
  await expect(conversation(page).getByText("Respuesta breve sobre: hola.")).toBeVisible();
  await page.getByRole("button", { name: "Volver" }).click();
  await page.goto("/chat");

  await list.getByRole("button", { name: "Eliminar «Mi análisis de Russafa»" }).click();
  await page.getByRole("dialog").getByRole("button", { name: "Eliminar" }).click();
  await expect(list.getByText("Tus conversaciones aparecerán aquí.")).toBeVisible();
  await page.reload();
  await expect(list.getByText("Tus conversaciones aparecerán aquí.")).toBeVisible();
});

test("second_user_cannot_see_first_users_conversation", async ({ page, browser }) => {
  await sendMessage(page, "pregunta privada de Ana");
  await waitForTurnEnd(page);
  await page.getByRole("button", { name: "Abrir a pantalla completa" }).click();
  await expect(page).toHaveURL(/\/chat\/[0-9a-f-]{36}$/);
  const url = page.url();
  await expect(blocks(page).first()).toBeVisible();

  // Luis, with his own cookie, sees nothing of Ana's and cannot open her conversation by URL.
  const luis = await newSignedInPage(browser, LUIS);
  await luis.page.goto("/");
  await expect(luis.page.getByRole("button", { name: "Menú de usuario" })).toBeVisible();
  await expect(luis.page.getByText("pregunta privada de Ana")).toHaveCount(0);
  await luis.page.goto("/chat");
  await expect(luis.page.getByText("Sin conversaciones todavía")).toBeVisible();
  await luis.page.goto(url);
  await expect(luis.page).toHaveURL(/\/chat$/); // an unknown conversation goes back to the list
  await expect(luis.page.getByText("pregunta privada de Ana")).toHaveCount(0);
  expect(luis.errors).toEqual([]);
  await luis.context.close();

  // Ana still has hers.
  await page.goto("/chat");
  await expect(page.getByRole("button", { name: "pregunta privada de Ana" }).first()).toBeVisible();
  expect(ANA.email).not.toBe(LUIS.email);
});
