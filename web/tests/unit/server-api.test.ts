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

  it("fetch_me_unknown_user_is_null", async () => {
    for (const status of [401, 404]) {
      fetchMock.mockResolvedValue(Response.json({ detail: "x" }, { status }));
      await expect(fetchMe("u1"), String(status)).resolves.toBeNull();
    }
  });

  it("fetch_me_throws_on_other_errors", async () => {
    fetchMock.mockResolvedValue(new Response("", { status: 503 }));
    await expect(fetchMe("u1")).rejects.toThrow();
  });
});
