import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { createConversation, getConversation, sendMessage, STORAGE_KEY } from "@/lib/client";

const fetchMock = vi.fn();

beforeEach(() => vi.stubGlobal("fetch", fetchMock));
afterEach(() => {
  fetchMock.mockReset();
  vi.unstubAllGlobals();
});

describe("client", () => {
  it("storage_key", () => {
    expect(STORAGE_KEY).toBe("rooster.conversationId");
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
