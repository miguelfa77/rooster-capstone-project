// @vitest-environment node
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { User } from "next-auth";
import type { JWT } from "next-auth/jwt";
import { jwtCallback, sessionCallback, signInCallback } from "@/lib/auth-callbacks";
import { isPublicPath, safeNext, signInPath } from "@/lib/auth-routing";

const fetchMock = vi.fn();
const UID = "6f1c2c1e-6c55-4c43-9a59-3f0b9e6c2a10";

beforeEach(() => {
  vi.stubEnv("ROOSTER_API_URL", "http://api.test/");
  vi.stubEnv("ROOSTER_API_KEY", "k");
  vi.stubEnv("ROOSTER_ALLOWED_EMAILS", "ana@fund.com");
  vi.stubGlobal("fetch", fetchMock);
});

afterEach(() => {
  fetchMock.mockReset();
  vi.unstubAllEnvs();
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

function googleUser(email: string | null): User {
  return { id: "google-sub-1", email, name: "Ana", image: "https://img.test/a.png" };
}

describe("auth callbacks", () => {
  it("disallowed_email_redirects_to_invitacion", async () => {
    expect(await signInCallback({ user: googleUser("intruso@x.es") })).toBe("/entrar/invitacion");
    expect(await signInCallback({ user: googleUser(null) })).toBe("/entrar/invitacion");
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("allowed_email_upserts_user_and_stores_uid", async () => {
    fetchMock.mockResolvedValue(
      Response.json({ id: UID, email: "ana@fund.com", name: "Ana", image: null, preferences: {} }),
    );
    const user = googleUser("Ana@Fund.com");
    expect(await signInCallback({ user })).toBe(true);

    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [url, init] = fetchMock.mock.calls[0];
    expect(String(url)).toBe("http://api.test/users");
    expect(init.method).toBe("PUT");
    const headers = new Headers(init.headers);
    expect(headers.get("x-rooster-key")).toBe("k");
    expect(headers.get("content-type")).toBe("application/json");
    expect(JSON.parse(init.body)).toEqual({ email: "Ana@Fund.com", name: "Ana", image: "https://img.test/a.png" });

    // Auth.js passes the same user object to the jwt callback on first token creation.
    const token = await jwtCallback({ token: { sub: "google-sub-1" } as JWT, user });
    expect(token.uid).toBe(UID);

    // Later session reads carry the uid forward without calling the API again.
    const again = await jwtCallback({ token });
    expect(again.uid).toBe(UID);
    expect(fetchMock).toHaveBeenCalledTimes(1);

    const session = await sessionCallback({
      session: { user: { email: "ana@fund.com" }, expires: "2099-01-01T00:00:00.000Z" } as never,
      token,
    });
    expect(session.user.id).toBe(UID);
  });

  it("upsert_failure_blocks_sign_in", async () => {
    const log = vi.spyOn(console, "error").mockImplementation(() => {});
    const failures: Array<() => Promise<Response>> = [
      () => Promise.resolve(new Response("boom", { status: 500 })),
      () => Promise.reject(new TypeError("fetch failed")),
      () => Promise.resolve(Response.json({ email: "ana@fund.com" })),
    ];
    for (const fail of failures) {
      fetchMock.mockImplementationOnce(fail);
      const user = googleUser("ana@fund.com");
      expect(await signInCallback({ user })).toBe("/entrar?error=servicio");
      const token = await jwtCallback({ token: {} as JWT, user });
      expect(token.uid).toBeUndefined();
    }
    expect(log).toHaveBeenCalled();
  });

  it("missing_api_config_blocks_sign_in", async () => {
    vi.spyOn(console, "error").mockImplementation(() => {});
    vi.stubEnv("ROOSTER_API_KEY", "");
    expect(await signInCallback({ user: googleUser("ana@fund.com") })).toBe("/entrar?error=servicio");
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("session_without_uid_has_no_user_id", async () => {
    const session = await sessionCallback({
      session: { user: { email: "ana@fund.com" }, expires: "2099-01-01T00:00:00.000Z" } as never,
      token: { sub: "google-sub-1" } as JWT,
    });
    expect(session.user.id).toBeUndefined();
  });
});

describe("auth routing", () => {
  it("safe_next_accepts_only_same_origin_relative_paths", () => {
    expect(safeNext("/conversaciones/1?x=2")).toBe("/conversaciones/1?x=2");
    expect(safeNext("/")).toBe("/");
    for (const bad of [undefined, null, "", "//evil.test", "https://evil.test/", "evil", "/\\evil.test", ["/a"]]) {
      expect(safeNext(bad as never)).toBe("/");
    }
  });

  it("public_paths", () => {
    for (const p of ["/entrar", "/entrar/invitacion", "/compartir/abc", "/api/auth/callback/google"]) {
      expect(isPublicPath(p)).toBe(true);
    }
    for (const p of ["/", "/entrarx", "/compartir", "/inteligencia", "/api/authx"]) {
      expect(isPublicPath(p)).toBe(false);
    }
  });

  it("sign_in_path_keeps_next", () => {
    expect(signInPath("/inteligencia", "?m=1")).toBe("/entrar?next=%2Finteligencia%3Fm%3D1");
    expect(signInPath("/", "")).toBe("/entrar?next=%2F");
  });
});
