// Browser-side API client. Talks only to the Next proxy under /api/rooster.
// Every call treats 401 as "signed out": it sends the browser to /entrar and throws.
import type { ConversationOut, Preferences, UserOut } from "./types";

const BASE = "/api/rooster";

export const STORAGE_KEY = "rooster.conversationId";

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

export async function patchPreferences(p: Partial<Preferences>): Promise<Preferences> {
  const res = await call("me/preferences", {
    method: "PATCH",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(p),
  });
  if (!res.ok) throw new Error(`patch preferences failed: HTTP ${res.status}`);
  return (await res.json()) as Preferences;
}

export async function createConversation(): Promise<string> {
  const res = await call("conversations", { method: "POST" });
  if (!res.ok) throw new Error(`create conversation failed: HTTP ${res.status}`);
  const body = (await res.json()) as { id: string };
  return body.id;
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
