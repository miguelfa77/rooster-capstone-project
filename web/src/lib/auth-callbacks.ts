// Auth.js callbacks as plain functions so they can be unit-tested without Auth.js.
import type { DefaultSession, Session, User } from "next-auth";
import type { JWT } from "next-auth/jwt";
import { isAllowed } from "@/lib/allowlist";

declare module "next-auth" {
  interface User {
    /** Rooster user id from PUT /users, set by signInCallback for the jwt callback. */
    roosterId?: string;
  }
  interface Session {
    user: { id?: string } & DefaultSession["user"];
  }
}

declare module "next-auth/jwt" {
  interface JWT {
    uid?: string;
  }
}

export const INVITE_ONLY_PATH = "/entrar/invitacion";
export const SERVICE_ERROR_PATH = "/entrar?error=servicio";

async function upsertUser(user: User): Promise<string | null> {
  const base = process.env.ROOSTER_API_URL?.replace(/\/+$/, "");
  const key = process.env.ROOSTER_API_KEY;
  if (!base || !key) {
    console.error("rooster auth: ROOSTER_API_URL and ROOSTER_API_KEY must both be set");
    return null;
  }
  try {
    const res = await fetch(`${base}/users`, {
      method: "PUT",
      headers: { "content-type": "application/json", "x-rooster-key": key },
      body: JSON.stringify({ email: user.email, name: user.name ?? null, image: user.image ?? null }),
      cache: "no-store",
    });
    if (!res.ok) {
      console.error(`rooster auth: PUT /users returned ${res.status}`);
      return null;
    }
    const body = (await res.json()) as { id?: unknown };
    if (typeof body.id !== "string" || !body.id) {
      console.error("rooster auth: PUT /users response has no id");
      return null;
    }
    return body.id;
  } catch (err) {
    console.error("rooster auth: PUT /users failed", err);
    return null;
  }
}

/**
 * Allowlist gate plus user upsert. The upsert runs here rather than in the jwt callback
 * because only signIn can turn a failure into a redirect of our choosing. Auth.js (JWT
 * sessions, no adapter) hands this same `user` object to the jwt callback that creates
 * the first token, which is how `roosterId` reaches it.
 */
export async function signInCallback({ user }: { user: User }): Promise<true | string> {
  if (!isAllowed(user.email)) return INVITE_ONLY_PATH;
  const id = await upsertUser(user);
  if (!id) return SERVICE_ERROR_PATH;
  user.roosterId = id;
  return true;
}

export async function jwtCallback({ token, user }: { token: JWT; user?: User }): Promise<JWT> {
  if (user?.roosterId) token.uid = user.roosterId;
  return token;
}

export async function sessionCallback({ session, token }: { session: Session; token: JWT }): Promise<Session> {
  if (token.uid) session.user.id = token.uid;
  else delete session.user.id;
  return session;
}
