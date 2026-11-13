import { act, renderHook, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { storageKey } from "@/lib/client";
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

type Route = (url: string, init?: RequestInit) => Response | Promise<Response>;

function routes(handlers: Record<string, Route>) {
  fetchMock.mockImplementation(async (url: string, init?: RequestInit) => {
    const key = `${init?.method ?? "GET"} ${url}`;
    const handler = handlers[key];
    if (!handler) throw new Error(`unexpected fetch ${key}`);
    return handler(url, init);
  });
}

const KEY = storageKey("u1");

/** Streams a test leaves open; closed after each test. */
const openStreams: ReadableStreamDefaultController<Uint8Array>[] = [];
function openStream(): { body: ReadableStream<Uint8Array>; emit: (e: object) => void; close: () => void } {
  const enc = new TextEncoder();
  let ctrl!: ReadableStreamDefaultController<Uint8Array>;
  const body = new ReadableStream<Uint8Array>({ start: (c) => void (ctrl = c) });
  openStreams.push(ctrl);
  return {
    body,
    emit: (e) => ctrl.enqueue(enc.encode(`event: ${(e as { type: string }).type}\ndata: ${JSON.stringify(e)}\n\n`)),
    close: () => ctrl.close(),
  };
}

beforeEach(() => {
  localStorage.clear();
  vi.stubGlobal("fetch", fetchMock);
});

afterEach(() => {
  for (const c of openStreams.splice(0)) {
    try {
      c.close();
    } catch {
      // already closed
    }
  }
  fetchMock.mockReset();
  vi.unstubAllGlobals();
});

describe("useChat", () => {
  it("restores_history_from_storage", async () => {
    localStorage.setItem(KEY, "c1");
    const conv: ConversationOut = { id: "c1", title: null, messages: storedMessages };
    routes({ "GET /api/rooster/conversations/c1": () => Response.json(conv) });

    const { result } = renderHook(() => useChat({ userId: "u1" }));
    await waitFor(() => expect(result.current.state.loading).toBe(false));
    expect(result.current.state.conversationId).toBe("c1");
    expect(result.current.state.messages).toEqual(storedMessages);
  });

  it("stale_id_404_cleared", async () => {
    localStorage.setItem(KEY, "gone");
    routes({ "GET /api/rooster/conversations/gone": () => new Response("", { status: 404 }) });

    const { result } = renderHook(() => useChat({ userId: "u1" }));
    await waitFor(() => expect(result.current.state.loading).toBe(false));
    expect(localStorage.getItem(KEY)).toBeNull();
    expect(result.current.state.conversationId).toBeNull();
    expect(result.current.state.messages).toEqual([]);
  });

  it("first_send_creates_and_persists_conversation", async () => {
    routes({
      "POST /api/rooster/conversations": () => Response.json({ id: "new" }, { status: 201 }),
      "POST /api/rooster/conversations/new/messages": () => sse([{ type: "done" }]),
      "GET /api/rooster/conversations/new": () => Response.json({ id: "new", messages: storedMessages }),
    });

    const { result } = renderHook(() => useChat({ userId: "u1" }));
    await waitFor(() => expect(result.current.state.loading).toBe(false));
    await act(() => result.current.send("hola"));

    expect(localStorage.getItem(KEY)).toBe("new");
    expect(result.current.state.conversationId).toBe("new");
    const post = fetchMock.mock.calls.find(([u]) => u === "/api/rooster/conversations/new/messages");
    expect(JSON.parse(post![1].body)).toEqual({ text: "hola" });
  });

  it("stream_without_done_fails", async () => {
    localStorage.setItem(KEY, "c1");
    routes({
      "GET /api/rooster/conversations/c1": () => Response.json({ id: "c1", messages: [] }),
      "POST /api/rooster/conversations/c1/messages": () => sse([{ type: "text_delta", text: "a medias" }]),
    });

    const { result } = renderHook(() => useChat({ userId: "u1" }));
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
    localStorage.setItem(KEY, "c1");
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

    const { result } = renderHook(() => useChat({ userId: "u1" }));
    await waitFor(() => expect(result.current.state.loading).toBe(false));
    await act(() => result.current.send("hola"));

    expect(gets).toBe(2);
    expect(result.current.state.messages).toEqual(storedMessages);
    expect(result.current.state.live).toBeNull();
  });

  it("409_sets_busy_keeps_text_and_polls_until_stored", async () => {
    localStorage.setItem(KEY, "c1");
    const grown: Message[] = [...storedMessages, { role: "user", content: "x" }, storedMessages[1]];
    let gets = 0;
    routes({
      "GET /api/rooster/conversations/c1": () =>
        Response.json({ id: "c1", messages: gets++ < 3 ? storedMessages : grown }),
      "POST /api/rooster/conversations/c1/messages": () => Response.json({ detail: "busy" }, { status: 409 }),
    });

    const { result } = renderHook(() => useChat({ userId: "u1" }));
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
    localStorage.setItem(KEY, "c1");
    let gets = 0;
    routes({
      "GET /api/rooster/conversations/c1": () => {
        gets++;
        return Response.json({ id: "c1", messages: storedMessages });
      },
      "POST /api/rooster/conversations/c1/messages": () => Response.json({ detail: "busy" }, { status: 409 }),
    });

    const { result } = renderHook(() => useChat({ userId: "u1" }));
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

  it("429_sets_rate_limited_keeps_text_and_does_not_poll", async () => {
    localStorage.setItem(KEY, "c1");
    let gets = 0;
    routes({
      "GET /api/rooster/conversations/c1": () => {
        gets++;
        return Response.json({ id: "c1", messages: storedMessages });
      },
      "POST /api/rooster/conversations/c1/messages": () =>
        Response.json({ detail: "rate_limited" }, { status: 429 }),
    });

    const { result } = renderHook(() => useChat({ userId: "u1" }));
    await waitFor(() => expect(result.current.state.loading).toBe(false));
    vi.useFakeTimers();
    try {
      await act(() => result.current.send("otra"));

      expect(result.current.state.notice).toBe(COPY.rateLimited);
      expect(result.current.state.draft).toEqual({ text: "otra" });
      expect(result.current.state.live).toBeNull();
      expect(result.current.state.messages).toEqual(storedMessages);
      await act(() => vi.advanceTimersByTimeAsync(5 * 60_000));
      expect(gets).toBe(1);
    } finally {
      vi.useRealTimers();
    }
  });

  it("error_event_skips_refetch_and_keeps_turn", async () => {
    localStorage.setItem(KEY, "c1");
    routes({
      "GET /api/rooster/conversations/c1": () => Response.json({ id: "c1", messages: [] }),
      "POST /api/rooster/conversations/c1/messages": () =>
        sse([{ type: "error", reason: "boom", suggestion: "otra vez" }, { type: "done" }]),
    });

    const { result } = renderHook(() => useChat({ userId: "u1" }));
    await waitFor(() => expect(result.current.state.loading).toBe(false));
    await act(() => result.current.send("hola"));

    expect(result.current.state.live?.status).toBe("done");
    expect(result.current.state.live?.error).toEqual({ reason: "boom", suggestion: "otra vez" });
    expect(fetchMock.mock.calls.filter(([, init]) => !init?.method)).toHaveLength(1);
  });

  it("refetch_failure_keeps_done_turn_without_error", async () => {
    localStorage.setItem(KEY, "c1");
    let gets = 0;
    routes({
      "GET /api/rooster/conversations/c1": () => {
        if (gets++ === 0) return Response.json({ id: "c1", messages: [] });
        throw new Error("network");
      },
      "POST /api/rooster/conversations/c1/messages": () =>
        sse([{ type: "text_delta", text: "respuesta" }, { type: "done" }]),
    });

    const { result } = renderHook(() => useChat({ userId: "u1" }));
    await waitFor(() => expect(result.current.state.loading).toBe(false));
    await act(() => result.current.send("hola"));

    expect(gets).toBe(2);
    expect(result.current.state.live?.status).toBe("done");
    expect(result.current.state.live?.text).toBe("respuesta");
    expect(result.current.state.live?.error).toBeNull();
  });

  it("open_and_reset_ignored_while_streaming", async () => {
    localStorage.setItem(KEY, "c1");
    const stream = openStream();
    routes({
      "GET /api/rooster/conversations/c1": () => Response.json({ id: "c1", messages: storedMessages }),
      "POST /api/rooster/conversations/c1/messages": () =>
        new Response(stream.body, { status: 200, headers: { "content-type": "text/event-stream" } }),
    });

    const { result } = renderHook(() => useChat({ userId: "u1" }));
    await waitFor(() => expect(result.current.state.loading).toBe(false));
    let pending!: Promise<void>;
    act(() => {
      pending = result.current.send("hola");
    });
    await waitFor(() => expect(result.current.state.live?.status).toBe("streaming"));

    act(() => void result.current.open("c2"));
    act(() => result.current.reset());
    await act(() => result.current.startChat("otra"));
    expect(result.current.state.conversationId).toBe("c1");
    expect(result.current.state.live?.userText).toBe("hola");
    expect(localStorage.getItem(KEY)).toBe("c1");
    expect(result.current.busy).toBe(true);

    await act(async () => {
      stream.emit({ type: "done" });
      stream.close();
      await pending;
    });
    expect(result.current.busy).toBe(false);
    // Nothing was fetched for c2 and only one message was posted.
    expect(fetchMock.mock.calls.map(([u]) => u)).not.toContain("/api/rooster/conversations/c2");
    expect(fetchMock.mock.calls.filter(([, init]) => init?.method === "POST")).toHaveLength(1);
  });

  it("open_loads_and_marks_active", async () => {
    localStorage.setItem(KEY, "c1");
    const other: Message[] = [{ role: "user", content: "otra" }];
    let resolveOpen!: (r: Response) => void;
    routes({
      "GET /api/rooster/conversations/c1": () => Response.json({ id: "c1", messages: storedMessages }),
    });
    const routed = fetchMock.getMockImplementation()!;
    fetchMock.mockImplementation((url: string, init?: RequestInit) =>
      url === "/api/rooster/conversations/c2"
        ? new Promise<Response>((r) => (resolveOpen = r))
        : routed(url, init),
    );

    const { result } = renderHook(() => useChat({ userId: "u1" }));
    await waitFor(() => expect(result.current.state.loading).toBe(false));
    act(() => void result.current.open("c2"));
    // Marked active (and remembered) right away; messages arrive with the load.
    expect(result.current.state.conversationId).toBe("c2");
    expect(result.current.state.loading).toBe(true);
    expect(result.current.state.messages).toEqual([]);
    expect(localStorage.getItem(KEY)).toBe("c2");

    await act(async () => resolveOpen(Response.json({ id: "c2", messages: other })));
    expect(result.current.state.loading).toBe(false);
    expect(result.current.state.messages).toEqual(other);

    // Re-opening the active conversation does not refetch it.
    const calls = fetchMock.mock.calls.length;
    act(() => void result.current.open("c2"));
    expect(fetchMock.mock.calls.length).toBe(calls);
  });

  it("open_404_clears_stored_id", async () => {
    routes({ "GET /api/rooster/conversations/gone": () => new Response("", { status: 404 }) });
    const onConversationsChanged = vi.fn();

    const { result } = renderHook(() => useChat({ userId: "u1", onConversationsChanged }));
    await waitFor(() => expect(result.current.state.loading).toBe(false));
    let found: boolean | undefined;
    await act(async () => {
      found = await result.current.open("gone");
    });
    expect(found).toBe(false);
    await waitFor(() => expect(result.current.state.loading).toBe(false));

    expect(result.current.state.conversationId).toBeNull();
    expect(localStorage.getItem(KEY)).toBeNull();
    expect(onConversationsChanged).toHaveBeenCalled();
  });

  it("storage_key_is_per_user", async () => {
    localStorage.setItem(KEY, "c1");
    routes({
      "POST /api/rooster/conversations": () => Response.json({ id: "mine" }, { status: 201 }),
      "POST /api/rooster/conversations/mine/messages": () => sse([{ type: "done" }]),
      "GET /api/rooster/conversations/mine": () => Response.json({ id: "mine", messages: storedMessages }),
    });

    // Another user on the same browser does not pick up u1's conversation.
    const { result } = renderHook(() => useChat({ userId: "u2" }));
    await waitFor(() => expect(result.current.state.loading).toBe(false));
    expect(result.current.state.conversationId).toBeNull();
    await act(() => result.current.send("hola"));

    expect(localStorage.getItem(storageKey("u2"))).toBe("mine");
    expect(localStorage.getItem(KEY)).toBe("c1");
    expect(fetchMock.mock.calls.map(([u]) => u)).not.toContain("/api/rooster/conversations/c1");
  });

  it("refresh_after_turn_finishes", async () => {
    let onCreatedCalls = -1;
    const onConversationsChanged = vi.fn();
    routes({
      "POST /api/rooster/conversations": () => Response.json({ id: "new" }, { status: 201 }),
      "POST /api/rooster/conversations/new/messages": () => {
        onCreatedCalls = onConversationsChanged.mock.calls.length;
        return sse([{ type: "done" }]);
      },
      "GET /api/rooster/conversations/new": () => Response.json({ id: "new", messages: storedMessages }),
    });

    const { result } = renderHook(() => useChat({ userId: "u1", onConversationsChanged }));
    await waitFor(() => expect(result.current.state.loading).toBe(false));
    await act(() => result.current.send("hola"));

    // Once after creating the conversation, once after the turn finished (the title may be set).
    expect(onCreatedCalls).toBe(1);
    expect(onConversationsChanged).toHaveBeenCalledTimes(2);
  });

  it("start_chat_resets_then_sends", async () => {
    localStorage.setItem(KEY, "c1");
    routes({
      "GET /api/rooster/conversations/c1": () => Response.json({ id: "c1", messages: storedMessages }),
      "POST /api/rooster/conversations": () => Response.json({ id: "new" }, { status: 201 }),
      "POST /api/rooster/conversations/new/messages": () => sse([{ type: "done" }]),
      "GET /api/rooster/conversations/new": () =>
        Response.json({ id: "new", messages: [{ role: "user", content: "pregunta" }] }),
    });

    const { result } = renderHook(() => useChat({ userId: "u1" }));
    await waitFor(() => expect(result.current.state.conversationId).toBe("c1"));
    await act(() => result.current.startChat("pregunta"));

    expect(result.current.state.conversationId).toBe("new");
    expect(result.current.state.messages).toEqual([{ role: "user", content: "pregunta" }]);
    expect(localStorage.getItem(KEY)).toBe("new");
    expect(fetchMock.mock.calls.map(([u]) => u)).not.toContain("/api/rooster/conversations/c1/messages");
  });

  it("mount_result_ignored_after_send_created_conversation", async () => {
    localStorage.setItem(KEY, "old");
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

    const { result } = renderHook(() => useChat({ userId: "u1" }));
    await act(() => result.current.send("hola"));
    await act(async () => {
      resolveMount(new Response("", { status: 404 }));
    });

    expect(localStorage.getItem(KEY)).toBe("new");
    expect(result.current.state.conversationId).toBe("new");
    expect(result.current.state.messages).toEqual(storedMessages);
  });

  it("concurrent_sends_post_once", async () => {
    localStorage.setItem(KEY, "c1");
    routes({
      "GET /api/rooster/conversations/c1": () => Response.json({ id: "c1", messages: [] }),
      "POST /api/rooster/conversations/c1/messages": () => sse([{ type: "done" }]),
    });

    const { result } = renderHook(() => useChat({ userId: "u1" }));
    await waitFor(() => expect(result.current.state.loading).toBe(false));
    await act(() => Promise.all([result.current.send("uno"), result.current.send("dos")]));

    const posts = fetchMock.mock.calls.filter(([u]) => u === "/api/rooster/conversations/c1/messages");
    expect(posts).toHaveLength(1);
    expect(JSON.parse(posts[0][1].body)).toEqual({ text: "uno" });
  });

  it("busy_covers_refetch_window", async () => {
    localStorage.setItem(KEY, "c1");
    let gets = 0;
    let resolveRefetch!: (r: Response) => void;
    routes({
      "GET /api/rooster/conversations/c1": () =>
        gets++ === 0
          ? Response.json({ id: "c1", messages: [] })
          : new Promise<Response>((r) => (resolveRefetch = r)),
      "POST /api/rooster/conversations/c1/messages": () => sse([{ type: "done" }]),
    });

    const { result } = renderHook(() => useChat({ userId: "u1" }));
    await waitFor(() => expect(result.current.state.loading).toBe(false));
    expect(result.current.busy).toBe(false);
    let pending!: Promise<void>;
    act(() => {
      pending = result.current.send("hola");
    });
    // The turn is done but its stored copy is still being fetched: still busy, still ignoring.
    await waitFor(() => expect(gets).toBe(2));
    expect(result.current.state.live?.status).toBe("done");
    expect(result.current.busy).toBe(true);
    act(() => void result.current.open("c2"));
    expect(result.current.state.conversationId).toBe("c1");

    await act(async () => {
      resolveRefetch(Response.json({ id: "c1", messages: storedMessages }));
      await pending;
    });
    expect(result.current.busy).toBe(false);
  });

  it("unmount_aborts_in_flight_send_and_makes_no_further_requests", async () => {
    localStorage.setItem(KEY, "c1");
    const stream = openStream();
    const onConversationsChanged = vi.fn();
    routes({
      "GET /api/rooster/conversations/c1": () => Response.json({ id: "c1", messages: [] }),
      "POST /api/rooster/conversations/c1/messages": () =>
        new Response(stream.body, { status: 200, headers: { "content-type": "text/event-stream" } }),
    });

    const { result, unmount } = renderHook(() => useChat({ userId: "u1", onConversationsChanged }));
    await waitFor(() => expect(result.current.state.loading).toBe(false));
    let pending!: Promise<void>;
    act(() => {
      pending = result.current.send("hola");
    });
    await waitFor(() => expect(result.current.state.live?.status).toBe("streaming"));
    const post = fetchMock.mock.calls.find(([u]) => u === "/api/rooster/conversations/c1/messages")!;
    const signal = (post[1] as RequestInit).signal!;
    expect(signal.aborted).toBe(false);

    unmount();
    expect(signal.aborted).toBe(true);
    const calls = fetchMock.mock.calls.length;
    await act(async () => {
      stream.emit({ type: "done" });
      stream.close();
      await pending;
    });
    expect(fetchMock.mock.calls.length).toBe(calls);
    expect(onConversationsChanged).not.toHaveBeenCalled();
  });

  it("409_after_unmount_does_not_poll", async () => {
    localStorage.setItem(KEY, "c1");
    let resolvePost!: (r: Response) => void;
    let gets = 0;
    routes({
      "GET /api/rooster/conversations/c1": () => {
        gets++;
        return Response.json({ id: "c1", messages: [] });
      },
      "POST /api/rooster/conversations/c1/messages": () => new Promise<Response>((r) => (resolvePost = r)),
    });

    const { result, unmount } = renderHook(() => useChat({ userId: "u1" }));
    await waitFor(() => expect(result.current.state.loading).toBe(false));
    vi.useFakeTimers();
    try {
      let pending!: Promise<void>;
      act(() => {
        pending = result.current.send("hola");
      });
      await act(() => vi.advanceTimersByTimeAsync(0));
      unmount();
      await act(async () => {
        resolvePost(Response.json({ detail: "busy" }, { status: 409 }));
        await pending;
      });
      await act(() => vi.advanceTimersByTimeAsync(60_000));
      expect(gets).toBe(1);
    } finally {
      vi.useRealTimers();
    }
  });
});
