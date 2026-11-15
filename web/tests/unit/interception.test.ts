// @vitest-environment node
// src/proxy.ts: Next 16 request interception (signed-out pages -> /entrar?next=...).
import { describe, expect, it, vi } from "vitest";
import { NextRequest } from "next/server";
import { unstable_doesMiddlewareMatch } from "next/experimental/testing/server";
import { config, proxy } from "@/proxy";

// auth(handler) wraps the handler and puts the session on request.auth; the test sets it directly.
vi.mock("@/auth", () => ({ auth: (handler: unknown) => handler }));

type Handler = (req: NextRequest & { auth: unknown }) => Response;

function run(url: string, session: unknown) {
  const req = Object.assign(new NextRequest(url), { auth: session });
  return (proxy as unknown as Handler)(req);
}

const signedIn = { user: { id: "6f1c2c1e-6c55-4c43-9a59-3f0b9e6c2a10" }, expires: "2099-01-01T00:00:00.000Z" };

describe("request interception", () => {
  it("signed_in_passes_through", () => {
    const res = run("http://web.test/inteligencia", signedIn);
    expect(res.headers.get("location")).toBeNull();
    expect(res.headers.get("x-middleware-next")).toBe("1");
  });

  it("public_paths_pass_through_signed_out", () => {
    for (const path of ["/entrar", "/entrar/invitacion?x=1", "/compartir/abc"]) {
      const res = run(`http://web.test${path}`, null);
      expect(res.headers.get("location")).toBeNull();
      expect(res.headers.get("x-middleware-next")).toBe("1");
    }
  });

  it("signed_out_redirects_to_entrar_with_next", () => {
    for (const session of [null, { user: {}, expires: "2099-01-01T00:00:00.000Z" }]) {
      const res = run("http://web.test/inteligencia?m=1", session);
      expect(res.status).toBe(307);
      expect(res.headers.get("location")).toBe("http://web.test/entrar?next=%2Finteligencia%3Fm%3D1");
    }
  });

  it("matcher_excludes_only_anchored_paths", () => {
    const matches = (url: string) => unstable_doesMiddlewareMatch({ config, url });
    for (const url of ["/api/auth/session", "/api/rooster/me", "/_next/static/x.js", "/_next/image?url=a", "/favicon.ico"]) {
      expect(matches(url), url).toBe(false);
    }
    for (const url of ["/", "/inteligencia", "/api/authX", "/api/roosterX/me", "/api/auth", "/favicon.icox", "/_next/staticX"]) {
      expect(matches(url), url).toBe(true);
    }
  });
});
