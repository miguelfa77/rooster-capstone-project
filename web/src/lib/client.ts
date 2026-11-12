// Browser-side API client. Talks only to the Next proxy under /api/rooster.
import type { ConversationOut } from "./types";

const BASE = "/api/rooster";

export const STORAGE_KEY = "rooster.conversationId";

export async function createConversation(): Promise<string> {
  const res = await fetch(`${BASE}/conversations`, { method: "POST" });
  if (!res.ok) throw new Error(`create conversation failed: HTTP ${res.status}`);
  const body = (await res.json()) as { id: string };
  return body.id;
}

/** Returns null when the conversation does not exist (404). */
export async function getConversation(id: string): Promise<ConversationOut | null> {
  const res = await fetch(`${BASE}/conversations/${encodeURIComponent(id)}`);
  if (res.status === 404) return null;
  if (!res.ok) throw new Error(`get conversation failed: HTTP ${res.status}`);
  return (await res.json()) as ConversationOut;
}

/** Posts a message; the caller inspects the status and reads the SSE body. */
export function sendMessage(id: string, text: string, signal?: AbortSignal): Promise<Response> {
  return fetch(`${BASE}/conversations/${encodeURIComponent(id)}/messages`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ text }),
    signal,
  });
}
