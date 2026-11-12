import { act, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ChatProvider, useChatContext } from "@/lib/chat-context";
import type { Message } from "@/lib/types";

const fetchMock = vi.fn();

const stored: Message[] = [
  { role: "user", content: "hola" },
  { role: "assistant", blocks: [], summary: "guardada", primitive_payloads: [], follow_ups: [] },
];

function Consumer() {
  const { state, send } = useChatContext();
  return (
    <div>
      <button type="button" onClick={() => void send("hola")}>
        enviar
      </button>
      <p data-testid="live">{state.live ? `${state.live.status}:${state.live.text}` : "none"}</p>
      <p data-testid="count">{state.messages.length}</p>
    </div>
  );
}

function Shell({ show }: { show: boolean }) {
  return <ChatProvider>{show ? <Consumer /> : null}</ChatProvider>;
}

beforeEach(() => {
  localStorage.clear();
  vi.stubGlobal("fetch", fetchMock);
});

afterEach(() => {
  fetchMock.mockReset();
  vi.unstubAllGlobals();
});

describe("ChatProvider", () => {
  it("live_turn_survives_consumer_remount", async () => {
    const enc = new TextEncoder();
    let stream!: ReadableStreamDefaultController<Uint8Array>;
    const body = new ReadableStream<Uint8Array>({ start: (c) => void (stream = c) });
    const emit = (e: { type: string; text?: string }) => stream.enqueue(enc.encode(`event: ${e.type}\ndata: ${JSON.stringify(e)}\n\n`));
    let gets = 0;
    fetchMock.mockImplementation(async (url: string, init?: RequestInit) => {
      const key = `${init?.method ?? "GET"} ${url}`;
      if (key === "POST /api/rooster/conversations") return Response.json({ id: "c1" }, { status: 201 });
      if (key === "POST /api/rooster/conversations/c1/messages")
        return new Response(body, { status: 200, headers: { "content-type": "text/event-stream" } });
      if (key === "GET /api/rooster/conversations/c1") {
        gets++;
        return Response.json({ id: "c1", messages: stored });
      }
      throw new Error(`unexpected fetch ${key}`);
    });

    const { rerender } = render(<Shell show />);
    act(() => screen.getByRole("button", { name: "enviar" }).click());
    await act(async () => emit({ type: "text_delta", text: "a medias" }));
    await waitFor(() => expect(screen.getByTestId("live")).toHaveTextContent("streaming:a medias"));

    // Navigating away unmounts the page; the provider (root layout) stays mounted.
    rerender(<Shell show={false} />);
    expect(screen.queryByTestId("live")).toBeNull();
    await act(async () => emit({ type: "text_delta", text: " y más" }));

    rerender(<Shell show />);
    await waitFor(() => expect(screen.getByTestId("live")).toHaveTextContent("streaming:a medias y más"));

    await act(async () => {
      emit({ type: "done" });
      stream.close();
    });
    await waitFor(() => expect(screen.getByTestId("live")).toHaveTextContent("none"));
    expect(screen.getByTestId("count")).toHaveTextContent("2");
    expect(gets).toBe(1);
  });
});
