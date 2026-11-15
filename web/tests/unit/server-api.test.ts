import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("server-only", () => ({}));
vi.mock("@/auth", () => ({ currentUserId: vi.fn() }));

const { fetchMe } = await import("@/lib/server-api");

const fetchMock = vi.fn();

beforeEach(() => {
  vi.stubGlobal("fetch", fetchMock);
  vi.stubEnv("ROOSTER_API_URL", "http://api.test/");
  vi.stubEnv("ROOSTER_API_KEY", "k");
});
afterEach(() => {
  fetchMock.mockReset();
  vi.unstubAllGlobals();
  vi.unstubAllEnvs();
});

describe("server-api", () => {
  it("fetch_me_sends_key_and_user", async () => {
    const user = { id: "u1", email: "a@b.es", name: null, image: null, preferences: {} };
    fetchMock.mockResolvedValue(Response.json(user));
    await expect(fetchMe("u1")).resolves.toEqual(user);
    expect(fetchMock).toHaveBeenCalledWith("http://api.test/me", {
      headers: { "x-rooster-key": "k", "x-rooster-user": "u1" },
      cache: "no-store",
    });
  });

  it("fetch_me_null_on_user_required", async () => {
    fetchMock.mockResolvedValue(Response.json({ detail: "user_required" }, { status: 401 }));
    await expect(fetchMe("u1")).resolves.toBeNull();
  });

  it("fetch_me_throws_on_bad_key", async () => {
    fetchMock.mockResolvedValue(Response.json({ detail: "unauthorized" }, { status: 401 }));
    await expect(fetchMe("u1")).rejects.toThrow();
  });

  it("fetch_me_throws_on_other_errors", async () => {
    for (const res of [new Response("", { status: 503 }), Response.json({ detail: "x" }, { status: 404 }), new Response("no json", { status: 401 })]) {
      fetchMock.mockResolvedValue(res);
      await expect(fetchMe("u1"), String(res.status)).rejects.toThrow();
    }
  });
});
