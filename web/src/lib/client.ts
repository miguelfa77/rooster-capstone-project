// Browser-side API client. Talks only to the Next proxy under /api/rooster.
// Every call treats 401 as "signed out": it sends the browser to /entrar and throws.
import type { ConversationOut, ConversationSummary, Preferences, StoredPreferences, UserOut } from "./types";

const BASE = "/api/rooster";

/** localStorage key of the user's active conversation id. */
export function storageKey(userId: string): string {
  return `rooster.conversationId.${userId}`;
}

export class SignedOutError extends Error {
  constructor() {
    super("signed out");
    this.name = "SignedOutError";
  }
}

async function call(path: string, init?: RequestInit): Promise<Response> {
  const res = await (init === undefined ? fetch(`${BASE}/${path}`) : fetch(`${BASE}/${path}`, init));
  if (res.status === 401) {
    // A full load on purpose: it drops all client state of the ended session.
    // eslint-disable-next-line @next/next/no-location-assign-relative-destination
    window.location.assign("/entrar");
    throw new SignedOutError();
  }
  return res;
}

export async function getMe(): Promise<UserOut> {
  const res = await call("me");
  if (!res.ok) throw new Error(`get me failed: HTTP ${res.status}`);
  return (await res.json()) as UserOut;
}

export async function patchPreferences(p: Partial<Preferences>): Promise<StoredPreferences> {
  const res = await call("me/preferences", {
    method: "PATCH",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(p),
  });
  if (!res.ok) throw new Error(`patch preferences failed: HTTP ${res.status}`);
  return (await res.json()) as StoredPreferences;
}

export async function createConversation(): Promise<string> {
  const res = await call("conversations", { method: "POST" });
  if (!res.ok) throw new Error(`create conversation failed: HTTP ${res.status}`);
  const body = (await res.json()) as { id: string };
  return body.id;
}

/** The user's conversations, most recently updated first. */
export async function listConversations(): Promise<ConversationSummary[]> {
  const res = await call("conversations");
  if (!res.ok) throw new Error(`list conversations failed: HTTP ${res.status}`);
  return (await res.json()) as ConversationSummary[];
}

export async function renameConversation(id: string, title: string): Promise<ConversationSummary> {
  const res = await call(`conversations/${encodeURIComponent(id)}`, {
    method: "PATCH",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ title }),
  });
  if (!res.ok) throw new Error(`rename conversation failed: HTTP ${res.status}`);
  return (await res.json()) as ConversationSummary;
}

/** A conversation that is already gone (404) counts as deleted. */
export async function deleteConversation(id: string): Promise<void> {
  const res = await call(`conversations/${encodeURIComponent(id)}`, { method: "DELETE" });
  if (!res.ok && res.status !== 404) throw new Error(`delete conversation failed: HTTP ${res.status}`);
}

/** Returns null when the conversation does not exist (404). */
export async function getConversation(id: string): Promise<ConversationOut | null> {
  const res = await call(`conversations/${encodeURIComponent(id)}`);
  if (res.status === 404) return null;
  if (!res.ok) throw new Error(`get conversation failed: HTTP ${res.status}`);
  return (await res.json()) as ConversationOut;
}

/** Posts a message; the caller inspects the status and reads the SSE body. */
export function sendMessage(id: string, text: string, signal?: AbortSignal): Promise<Response> {
  return call(`conversations/${encodeURIComponent(id)}/messages`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ text }),
    signal,
  });
}

/** GET an /intel/* resource (path includes any query string); throws on a non-2xx status. */
export async function getIntel<T>(path: string, signal?: AbortSignal): Promise<T> {
  const res = await call(`intel/${path}`, signal ? { signal } : undefined);
  if (!res.ok) throw new Error(`intel ${path} failed: HTTP ${res.status}`);
  return (await res.json()) as T;
}
