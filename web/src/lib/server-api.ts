// Server-only calls to FastAPI with the shared key and the session's user id.
// Never import this from a client component.
import "server-only";
import { cache } from "react";
import { currentUserId } from "@/auth";
import type { UserOut } from "./types";

/**
 * GET /me for `userId`; null only when the API does not know the user (401 "user_required").
 * Anything else, including 401 "unauthorized" (wrong API key), throws: a misconfiguration
 * must surface as an error, not sign everyone out.
 */
export async function fetchMe(userId: string): Promise<UserOut | null> {
  const base = process.env.ROOSTER_API_URL?.replace(/\/+$/, "");
  const key = process.env.ROOSTER_API_KEY;
  if (!base || !key) throw new Error("ROOSTER_API_URL and ROOSTER_API_KEY must both be set");
  const res = await fetch(`${base}/me`, {
    headers: { "x-rooster-key": key, "x-rooster-user": userId },
    cache: "no-store",
  });
  if (res.status === 401 && (await detail(res)) === "user_required") return null;
  if (!res.ok) throw new Error(`GET /me failed: HTTP ${res.status}`);
  return (await res.json()) as UserOut;
}

async function detail(res: Response): Promise<unknown> {
  try {
    return ((await res.json()) as { detail?: unknown }).detail;
  } catch {
    return undefined;
  }
}

/** The signed-in user as the API knows them, or null; deduplicated within one request. */
export const getCurrentUser = cache(async (): Promise<UserOut | null> => {
  const id = await currentUserId();
  return id ? fetchMe(id) : null;
});
