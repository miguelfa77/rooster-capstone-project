import { act, renderHook, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { STORAGE_KEY } from "@/lib/client";
import { COPY } from "@/lib/copy";
import { useChat } from "@/lib/use-chat";
import type { ConversationOut, Message } from "@/lib/types";

const fetchMock = vi.fn();

function sse(events: object[]): Response {
  const body = events.map((e) => `event: ${(e as { type: string }).type}\ndata: ${JSON.stringify(e)}\n\n`).join("");
  return new Response(body, { status: 200, headers: { "content-type": "text/event-stream" } });
}

const storedMessages: Message[] = [
  { role: "user", content: "hola" },
  { role: "assistant", blocks: [], summary: "respuesta", primitive_payloads: [], follow_ups: [] },
];

type Route = (url: string, init?: RequestInit) => Response;

function routes(handlers: Record<string, Route>) {
  fetchMock.mockImplementation(async (url: string, init?: RequestInit) => {
    const key = `${init?.method ?? "GET"} ${url}`;
    const handler = handlers[key];
    if (!handler) throw new Error(`unexpected fetch ${key}`);
    return handler(url, init);
  });
}

beforeEach(() => {
  localStorage.clear();
  vi.stubGlobal("fetch", fetchMock);
});

afterEach(() => {
  fetchMock.mockReset();
  vi.unstubAllGlobals();
});

describe("useChat", () => {
  it("restores_history_from_storage", async () => {
    localStorage.setItem(STORAGE_KEY, "c1");
    const conv: ConversationOut = { id: "c1", messages: storedMessages };
    routes({ "GET /api/rooster/conversations/c1": () => Response.json(conv) });

    const { result } = renderHook(() => useChat());
    await waitFor(() => expect(result.current.state.loading).toBe(false));
    expect(result.current.state.conversationId).toBe("c1");
    expect(result.current.state.messages).toEqual(storedMessages);
  });

  it("stale_id_cleared_on_404", async () => {
    localStorage.setItem(STORAGE_KEY, "gone");
    routes({ "GET /api/rooster/conversations/gone": () => new Response("", { status: 404 }) });

    const { result } = renderHook(() => useChat());
    await waitFor(() => expect(result.current.state.loading).toBe(false));
    expect(localStorage.getItem(STORAGE_KEY)).toBeNull();
    expect(result.current.state.conversationId).toBeNull();
    expect(result.current.state.messages).toEqual([]);
  });

  it("first_send_creates_and_persists_conversation", async () => {
    routes({
      "POST /api/rooster/conversations": () => Response.json({ id: "new" }, { status: 201 }),
      "POST /api/rooster/conversations/new/messages": () => sse([{ type: "done" }]),
      "GET /api/rooster/conversations/new": () => Response.json({ id: "new", messages: storedMessages }),
    });

    const { result } = renderHook(() => useChat());
    await waitFor(() => expect(result.current.state.loading).toBe(false));
    await act(() => result.current.send("hola"));

    expect(localStorage.getItem(STORAGE_KEY)).toBe("new");
    expect(result.current.state.conversationId).toBe("new");
    const post = fetchMock.mock.calls.find(([u]) => u === "/api/rooster/conversations/new/messages");
    expect(JSON.parse(post![1].body)).toEqual({ text: "hola" });
  });

  it("stream_without_done_fails", async () => {
    localStorage.setItem(STORAGE_KEY, "c1");
    routes({
      "GET /api/rooster/conversations/c1": () => Response.json({ id: "c1", messages: [] }),
      "POST /api/rooster/conversations/c1/messages": () => sse([{ type: "text_delta", text: "a medias" }]),
    });

    const { result } = renderHook(() => useChat());
    await waitFor(() => expect(result.current.state.loading).toBe(false));
    await act(() => result.current.send("hola"));

    const live = result.current.state.live;
    expect(live?.status).toBe("done");
    expect(live?.text).toBe("a medias");
    expect(live?.error?.suggestion).toBe(COPY.chatErrorSuggestion);
    expect(live?.error?.reason).toBe(COPY.chatStreamInterrupted);
    // Only the initial GET: no refetch after an incomplete stream.
    expect(fetchMock.mock.calls.filter(([, init]) => !init?.method)).toHaveLength(1);
  });

  it("refetches_after_done", async () => {
    localStorage.setItem(STORAGE_KEY, "c1");
    let gets = 0;
    routes({
      "GET /api/rooster/conversations/c1": () =>
        Response.json({ id: "c1", messages: gets++ === 0 ? [] : storedMessages }),
      "POST /api/rooster/conversations/c1/messages": () =>
        sse([
          { type: "thought", text: "pienso" },
          { type: "text_delta", text: "respuesta" },
          { type: "done" },
        ]),
    });

    const { result } = renderHook(() => useChat());
    await waitFor(() => expect(result.current.state.loading).toBe(false));
    await act(() => result.current.send("hola"));

    expect(gets).toBe(2);
    expect(result.current.state.messages).toEqual(storedMessages);
    expect(result.current.state.live).toBeNull();
  });

  it("409_sets_busy_keeps_text_and_polls_until_stored", async () => {
    localStorage.setItem(STORAGE_KEY, "c1");
    const grown: Message[] = [...storedMessages, { role: "user", content: "x" }, storedMessages[1]];
    let gets = 0;
    routes({
      "GET /api/rooster/conversations/c1": () =>
        Response.json({ id: "c1", messages: gets++ < 3 ? storedMessages : grown }),
      "POST /api/rooster/conversations/c1/messages": () => Response.json({ detail: "busy" }, { status: 409 }),
    });

    const { result } = renderHook(() => useChat());
    await waitFor(() => expect(result.current.state.loading).toBe(false));
    vi.useFakeTimers();
    try {
      await act(() => result.current.send("otra"));

      expect(result.current.state.notice).toBe(COPY.busy);
      expect(result.current.state.draft).toEqual({ text: "otra" });
      expect(result.current.state.live).toBeNull();
      expect(result.current.state.messages).toEqual(storedMessages);
      expect(gets).toBe(1);

      // Polls every 3 s; the first two polls see no new turn.
      await act(() => vi.advanceTimersByTimeAsync(3000));
      expect(gets).toBe(2);
      await act(() => vi.advanceTimersByTimeAsync(3000));
      expect(gets).toBe(3);
      expect(result.current.state.notice).toBe(COPY.busy);
      await act(() => vi.advanceTimersByTimeAsync(3000));
      expect(gets).toBe(4);
      expect(result.current.state.messages).toEqual(grown);
      expect(result.current.state.notice).toBeNull();

      // Polling stopped once the turn was stored.
      await act(() => vi.advanceTimersByTimeAsync(30_000));
      expect(gets).toBe(4);
    } finally {
      vi.useRealTimers();
    }
  });

  it("409_poll_gives_up_after_two_minutes", async () => {
    localStorage.setItem(STORAGE_KEY, "c1");
    let gets = 0;
    routes({
      "GET /api/rooster/conversations/c1": () => {
        gets++;
        return Response.json({ id: "c1", messages: storedMessages });
      },
      "POST /api/rooster/conversations/c1/messages": () => Response.json({ detail: "busy" }, { status: 409 }),
    });

    const { result } = renderHook(() => useChat());
    await waitFor(() => expect(result.current.state.loading).toBe(false));
    vi.useFakeTimers();
    try {
      await act(() => result.current.send("otra"));
      await act(() => vi.advanceTimersByTimeAsync(10 * 60_000));
      expect(gets).toBe(1 + 40);
    } finally {
      vi.useRealTimers();
    }
  });

  it("error_event_skips_refetch_and_keeps_turn", async () => {
    localStorage.setItem(STORAGE_KEY, "c1");
    routes({
      "GET /api/rooster/conversations/c1": () => Response.json({ id: "c1", messages: [] }),
      "POST /api/rooster/conversations/c1/messages": () =>
        sse([{ type: "error", reason: "boom", suggestion: "otra vez" }, { type: "done" }]),
    });

    const { result } = renderHook(() => useChat());
    await waitFor(() => expect(result.current.state.loading).toBe(false));
    await act(() => result.current.send("hola"));

    expect(result.current.state.live?.status).toBe("done");
    expect(result.current.state.live?.error).toEqual({ reason: "boom", suggestion: "otra vez" });
    expect(fetchMock.mock.calls.filter(([, init]) => !init?.method)).toHaveLength(1);
  });

  it("refetch_failure_keeps_done_turn_without_error", async () => {
    localStorage.setItem(STORAGE_KEY, "c1");
    let gets = 0;
    routes({
      "GET /api/rooster/conversations/c1": () => {
        if (gets++ === 0) return Response.json({ id: "c1", messages: [] });
        throw new Error("network");
      },
      "POST /api/rooster/conversations/c1/messages": () =>
        sse([{ type: "text_delta", text: "respuesta" }, { type: "done" }]),
    });

    const { result } = renderHook(() => useChat());
    await waitFor(() => expect(result.current.state.loading).toBe(false));
    await act(() => result.current.send("hola"));

    expect(gets).toBe(2);
    expect(result.current.state.live?.status).toBe("done");
    expect(result.current.state.live?.text).toBe("respuesta");
    expect(result.current.state.live?.error).toBeNull();
  });

  it("reset_during_create_does_not_persist", async () => {
    let resolveCreate!: (r: Response) => void;
    fetchMock.mockImplementationOnce(() => new Promise<Response>((r) => (resolveCreate = r)));
    fetchMock.mockRejectedValue(new Error("no further fetch expected"));

    const { result } = renderHook(() => useChat());
    await waitFor(() => expect(result.current.state.loading).toBe(false));
    let pending!: Promise<void>;
    act(() => {
      pending = result.current.send("hola");
    });
    act(() => result.current.reset());
    await act(async () => {
      resolveCreate(Response.json({ id: "new" }, { status: 201 }));
      await pending;
    });

    expect(localStorage.getItem(STORAGE_KEY)).toBeNull();
    expect(result.current.state.conversationId).toBeNull();
    expect(result.current.state.live).toBeNull();
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it("mount_result_ignored_after_send_created_conversation", async () => {
    localStorage.setItem(STORAGE_KEY, "old");
    let resolveMount!: (r: Response) => void;
    routes({
      "POST /api/rooster/conversations": () => Response.json({ id: "new" }, { status: 201 }),
      "POST /api/rooster/conversations/new/messages": () => sse([{ type: "done" }]),
      "GET /api/rooster/conversations/new": () => Response.json({ id: "new", messages: storedMessages }),
    });
    const routed = fetchMock.getMockImplementation()!;
    fetchMock.mockImplementation((url: string, init?: RequestInit) =>
      url === "/api/rooster/conversations/old"
        ? new Promise<Response>((r) => (resolveMount = r))
        : routed(url, init),
    );

    const { result } = renderHook(() => useChat());
    await act(() => result.current.send("hola"));
    await act(async () => {
      resolveMount(new Response("", { status: 404 }));
    });

    expect(localStorage.getItem(STORAGE_KEY)).toBe("new");
    expect(result.current.state.conversationId).toBe("new");
    expect(result.current.state.messages).toEqual(storedMessages);
  });

  it("concurrent_sends_post_once", async () => {
    localStorage.setItem(STORAGE_KEY, "c1");
    routes({
      "GET /api/rooster/conversations/c1": () => Response.json({ id: "c1", messages: [] }),
      "POST /api/rooster/conversations/c1/messages": () => sse([{ type: "done" }]),
    });

    const { result } = renderHook(() => useChat());
    await waitFor(() => expect(result.current.state.loading).toBe(false));
    await act(() => Promise.all([result.current.send("uno"), result.current.send("dos")]));

    const posts = fetchMock.mock.calls.filter(([u]) => u === "/api/rooster/conversations/c1/messages");
    expect(posts).toHaveLength(1);
    expect(JSON.parse(posts[0][1].body)).toEqual({ text: "uno" });
  });
});
