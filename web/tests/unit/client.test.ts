import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  createConversation,
  deleteConversation,
  getConversation,
  getIntel,
  getMe,
  listConversations,
  patchPreferences,
  renameConversation,
  sendMessage,
  SignedOutError,
  storageKey,
} from "@/lib/client";

const fetchMock = vi.fn();

beforeEach(() => vi.stubGlobal("fetch", fetchMock));
afterEach(() => {
  fetchMock.mockReset();
  vi.unstubAllGlobals();
});

describe("client", () => {
  it("storage_key_is_per_user", () => {
    expect(storageKey("u1")).toBe("rooster.conversationId.u1");
    expect(storageKey("u2")).toBe("rooster.conversationId.u2");
  });

  it("list_conversations_returns_summaries", async () => {
    const items = [{ id: "c1", title: null, updated_at: "2026-11-13T10:00:00Z" }];
    fetchMock.mockResolvedValue(Response.json(items));
    await expect(listConversations()).resolves.toEqual(items);
    expect(fetchMock).toHaveBeenCalledWith("/api/rooster/conversations");

    fetchMock.mockResolvedValue(new Response("", { status: 500 }));
    await expect(listConversations()).rejects.toThrow();
  });

  it("rename_conversation_patches_title", async () => {
    const summary = { id: "c1", title: "Ruzafa", updated_at: "2026-11-13T10:00:00Z" };
    fetchMock.mockResolvedValue(Response.json(summary));
    await expect(renameConversation("c1", "Ruzafa")).resolves.toEqual(summary);
    expect(fetchMock).toHaveBeenCalledWith("/api/rooster/conversations/c1", {
      method: "PATCH",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ title: "Ruzafa" }),
    });

    fetchMock.mockResolvedValue(Response.json({ detail: "too long" }, { status: 422 }));
    await expect(renameConversation("c1", "x")).rejects.toThrow();
  });

  it("delete_conversation_sends_delete", async () => {
    fetchMock.mockResolvedValue(new Response(null, { status: 204 }));
    await expect(deleteConversation("c1")).resolves.toBeUndefined();
    expect(fetchMock).toHaveBeenCalledWith("/api/rooster/conversations/c1", { method: "DELETE" });

    // Already gone counts as deleted.
    fetchMock.mockResolvedValue(new Response("", { status: 404 }));
    await expect(deleteConversation("c1")).resolves.toBeUndefined();

    fetchMock.mockResolvedValue(new Response("", { status: 500 }));
    await expect(deleteConversation("c1")).rejects.toThrow();
  });

  it("create_conversation_posts_and_returns_id", async () => {
    fetchMock.mockResolvedValue(Response.json({ id: "c1" }, { status: 201 }));
    await expect(createConversation()).resolves.toBe("c1");
    expect(fetchMock).toHaveBeenCalledWith("/api/rooster/conversations", { method: "POST" });
  });

  it("create_conversation_throws_on_error", async () => {
    fetchMock.mockResolvedValue(new Response("", { status: 500 }));
    await expect(createConversation()).rejects.toThrow();
  });

  it("get_conversation_404_is_null", async () => {
    fetchMock.mockResolvedValue(new Response("", { status: 404 }));
    await expect(getConversation("c1")).resolves.toBeNull();
    expect(fetchMock).toHaveBeenCalledWith("/api/rooster/conversations/c1");
  });

  it("get_conversation_returns_body", async () => {
    fetchMock.mockResolvedValue(Response.json({ id: "c1", messages: [] }));
    await expect(getConversation("c1")).resolves.toEqual({ id: "c1", messages: [] });
  });

  it("send_message_posts_json_and_returns_response", async () => {
    const res = new Response("", { status: 200 });
    fetchMock.mockResolvedValue(res);
    const signal = new AbortController().signal;
    await expect(sendMessage("c1", "hola", signal)).resolves.toBe(res);
    expect(fetchMock).toHaveBeenCalledWith("/api/rooster/conversations/c1/messages", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ text: "hola" }),
      signal,
    });
  });
});

describe("client signed out", () => {
  const assign = vi.fn();
  beforeEach(() => vi.stubGlobal("location", { ...window.location, assign }));
  afterEach(() => assign.mockReset());

  it("client_401_redirects_to_entrar", async () => {
    const calls: [string, () => Promise<unknown>][] = [
      ["getMe", () => getMe()],
      ["patchPreferences", () => patchPreferences({ theme: "dark" })],
      ["createConversation", () => createConversation()],
      ["getConversation", () => getConversation("c1")],
      ["listConversations", () => listConversations()],
      ["renameConversation", () => renameConversation("c1", "t")],
      ["deleteConversation", () => deleteConversation("c1")],
      ["sendMessage", () => sendMessage("c1", "hola")],
      ["getIntel", () => getIntel("framing")],
    ];
    for (const [name, call] of calls) {
      assign.mockReset();
      fetchMock.mockResolvedValue(Response.json({ detail: "signed_out" }, { status: 401 }));
      await expect(call(), name).rejects.toThrow(SignedOutError);
      expect(assign, name).toHaveBeenCalledWith("/entrar");
    }
  });

  it("get_me_and_patch_preferences", async () => {
    fetchMock.mockResolvedValueOnce(Response.json({ id: "u1", email: "a@b.es", name: null, image: null, preferences: {} }));
    await expect(getMe()).resolves.toMatchObject({ id: "u1" });
    expect(fetchMock).toHaveBeenLastCalledWith("/api/rooster/me");

    fetchMock.mockResolvedValueOnce(Response.json({ theme: "dark" }));
    await expect(patchPreferences({ theme: "dark" })).resolves.toEqual({ theme: "dark" });
    expect(fetchMock).toHaveBeenLastCalledWith("/api/rooster/me/preferences", {
      method: "PATCH",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ theme: "dark" }),
    });

    fetchMock.mockResolvedValueOnce(new Response("", { status: 500 }));
    await expect(patchPreferences({ theme: "dark" })).rejects.toThrow();
    expect(assign).not.toHaveBeenCalled();
  });
});
