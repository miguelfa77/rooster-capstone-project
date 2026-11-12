// @vitest-environment node
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { DELETE, GET, PATCH, POST } from "@/app/api/rooster/[...path]/route";
import { currentUserId } from "@/auth";

vi.mock("@/auth", () => ({ currentUserId: vi.fn() }));

const fetchMock = vi.fn();
const UID = "6f1c2c1e-6c55-4c43-9a59-3f0b9e6c2a10";

function ctx(path: string[]) {
  return { params: Promise.resolve({ path }) };
}

beforeEach(() => {
  vi.stubEnv("ROOSTER_API_URL", "http://api.test");
  vi.stubEnv("ROOSTER_API_KEY", "k");
  vi.stubGlobal("fetch", fetchMock);
  vi.mocked(currentUserId).mockResolvedValue(UID);
});

afterEach(() => {
  fetchMock.mockReset();
  vi.unstubAllEnvs();
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
  vi.mocked(currentUserId).mockReset();
});

describe("rooster proxy", () => {
  it("forwards_with_key", async () => {
    fetchMock.mockResolvedValue(new Response("{}", { status: 200 }));
    const req = new Request("http://web.test/api/rooster/conversations/x/messages", {
      method: "POST",
      headers: {
        "content-type": "application/json",
        "x-rooster-key": "forged",
        "x-rooster-user": "someone-else",
        accept: "text/event-stream",
        "accept-language": "es-ES",
        "x-forwarded-for": "1.2.3.4",
        authorization: "Bearer t",
        cookie: "s=1",
        host: "web.test",
      },
      body: JSON.stringify({ text: "hola" }),
    });
    await POST(req, ctx(["conversations", "x", "messages"]));

    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [url, init] = fetchMock.mock.calls[0];
    expect(String(url)).toBe("http://api.test/conversations/x/messages");
    expect(init.method).toBe("POST");
    const headers = new Headers(init.headers);
    expect(headers.get("x-rooster-key")).toBe("k");
    expect(headers.get("content-type")).toBe("application/json");
    expect(headers.get("accept")).toBe("text/event-stream");
    expect(headers.get("accept-language")).toBe("es-ES");
    expect(headers.get("x-rooster-user")).toBe(UID);
    expect([...headers.keys()].sort()).toEqual([
      "accept",
      "accept-language",
      "content-type",
      "x-rooster-key",
      "x-rooster-user",
    ]);
    expect(new TextDecoder().decode(init.body)).toBe(JSON.stringify({ text: "hola" }));
    expect(init.signal).toBe(req.signal);
  });

  it("streams_body_through", async () => {
    const body = new ReadableStream<Uint8Array>({
      start(c) {
        c.enqueue(new TextEncoder().encode('data: {"type":"done"}\n\n'));
        c.close();
      },
    });
    fetchMock.mockResolvedValue(
      new Response(body, { status: 200, headers: { "content-type": "text/event-stream" } }),
    );
    const res = await POST(
      new Request("http://web.test/api/rooster/conversations/x/messages", { method: "POST", body: "{}" }),
      ctx(["conversations", "x", "messages"]),
    );
    expect(res.body).toBe(body);
    expect(res.headers.get("content-type")).toBe("text/event-stream");
    expect(res.headers.get("cache-control")).toBe("no-cache");
    expect(await res.text()).toBe('data: {"type":"done"}\n\n');
  });

  it("passes_status", async () => {
    fetchMock.mockResolvedValue(
      new Response('{"detail":"not_found"}', { status: 404, headers: { "content-type": "application/json" } }),
    );
    const res = await GET(new Request("http://web.test/api/rooster/conversations/nope"), ctx(["conversations", "nope"]));
    expect(res.status).toBe(404);
    expect(await res.json()).toEqual({ detail: "not_found" });
  });

  it("rejects_unknown_prefix", async () => {
    for (const path of [["health"], ["..", "x"], ["admin"], ["conversations", "..", "health"]]) {
      const res = await GET(new Request(`http://web.test/api/rooster/${path.join("/")}`), ctx(path));
      expect(res.status).toBe(404);
    }
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("upstream_down_502", async () => {
    const log = vi.spyOn(console, "error").mockImplementation(() => {});
    fetchMock.mockRejectedValue(new TypeError("fetch failed"));
    const res = await GET(new Request("http://web.test/api/rooster/intel/framing"), ctx(["intel", "framing"]));
    expect(res.status).toBe(502);
    expect(await res.json()).toEqual({ detail: "upstream_unavailable" });
    expect(log).toHaveBeenCalledTimes(1);
  });

  it("client_abort_not_logged", async () => {
    const log = vi.spyOn(console, "error").mockImplementation(() => {});
    const controller = new AbortController();
    controller.abort();
    fetchMock.mockRejectedValue(new DOMException("aborted", "AbortError"));
    const req = new Request("http://web.test/api/rooster/intel/framing", { signal: controller.signal });
    const res = await GET(req, ctx(["intel", "framing"]));
    expect(res.status).toBe(502);
    expect(log).not.toHaveBeenCalled();
  });

  it("misconfigured_500", async () => {
    const log = vi.spyOn(console, "error").mockImplementation(() => {});
    for (const missing of ["ROOSTER_API_URL", "ROOSTER_API_KEY", "ROOSTER_API_URL"]) {
      vi.stubEnv("ROOSTER_API_URL", "http://api.test");
      vi.stubEnv("ROOSTER_API_KEY", "k");
      vi.stubEnv(missing, "");
      const res = await GET(new Request("http://web.test/api/rooster/intel/framing"), ctx(["intel", "framing"]));
      expect(res.status).toBe(500);
      expect(await res.json()).toEqual({ detail: "proxy_misconfigured" });
    }
    expect(fetchMock).not.toHaveBeenCalled();
    expect(log).toHaveBeenCalledTimes(1);
  });

  it("query_string_forwarded", async () => {
    fetchMock.mockResolvedValue(new Response("[]", { status: 200 }));
    await GET(
      new Request("http://web.test/api/rooster/intel/neighborhoods?metric=a"),
      ctx(["intel", "neighborhoods"]),
    );
    expect(String(fetchMock.mock.calls[0][0])).toBe("http://api.test/intel/neighborhoods?metric=a");
    expect(fetchMock.mock.calls[0][1].method).toBe("GET");
    expect(fetchMock.mock.calls[0][1].body).toBeUndefined();
  });

  it("no_session_401_signed_out_without_upstream_call", async () => {
    vi.mocked(currentUserId).mockResolvedValue(null);
    for (const [handler, method, path] of [
      [GET, "GET", ["me"]],
      [POST, "POST", ["conversations", "x", "messages"]],
      [GET, "GET", ["intel", "framing"]],
    ] as const) {
      const res = await handler(
        new Request(`http://web.test/api/rooster/${path.join("/")}`, {
          method,
          headers: { "x-rooster-user": UID },
          body: method === "GET" ? undefined : "{}",
        }),
        ctx([...path]),
      );
      expect(res.status).toBe(401);
      expect(await res.json()).toEqual({ detail: "signed_out" });
    }
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("forwards_user_id_from_session_not_from_request", async () => {
    fetchMock.mockResolvedValue(new Response("[]", { status: 200 }));
    await GET(
      new Request("http://web.test/api/rooster/conversations", { headers: { "x-rooster-user": "forged-id" } }),
      ctx(["conversations"]),
    );
    expect(new Headers(fetchMock.mock.calls[0][1].headers).get("x-rooster-user")).toBe(UID);
  });

  it("me_prefix_and_patch_delete_forwarded", async () => {
    fetchMock.mockImplementation(() => Promise.resolve(new Response("{}", { status: 200 })));
    const patchBody = JSON.stringify({ theme: "dark" });
    await PATCH(
      new Request("http://web.test/api/rooster/me/preferences", {
        method: "PATCH",
        headers: { "content-type": "application/json" },
        body: patchBody,
      }),
      ctx(["me", "preferences"]),
    );
    const del = await DELETE(
      new Request("http://web.test/api/rooster/conversations/c1", { method: "DELETE" }),
      ctx(["conversations", "c1"]),
    );
    await GET(new Request("http://web.test/api/rooster/me"), ctx(["me"]));

    expect(fetchMock).toHaveBeenCalledTimes(3);
    const [[patchUrl, patchInit], [delUrl, delInit], [meUrl, meInit]] = fetchMock.mock.calls;
    expect(String(patchUrl)).toBe("http://api.test/me/preferences");
    expect(patchInit.method).toBe("PATCH");
    expect(new TextDecoder().decode(patchInit.body)).toBe(patchBody);
    expect(new Headers(patchInit.headers).get("x-rooster-user")).toBe(UID);
    expect(String(delUrl)).toBe("http://api.test/conversations/c1");
    expect(delInit.method).toBe("DELETE");
    expect(del.status).toBe(200);
    expect(String(meUrl)).toBe("http://api.test/me");
    expect(meInit.method).toBe("GET");
  });
});
