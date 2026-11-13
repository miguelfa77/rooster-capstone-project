import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ConversationList } from "@/components/chat/ConversationList";
import { ExamplePrompts } from "@/components/chat/ExamplePrompts";
import { ChatProvider, useChatContext } from "@/lib/chat-context";
import { storageKey } from "@/lib/client";
import { ConversationsProvider } from "@/lib/use-conversations";
import type { ConversationSummary } from "@/lib/types";

const router = vi.hoisted(() => ({ push: vi.fn() }));
vi.mock("next/navigation", () => ({ useRouter: () => router }));
const toast = vi.hoisted(() => ({ error: vi.fn() }));
vi.mock("sonner", () => ({ toast }));

const fetchMock = vi.fn();
const KEY = storageKey("u1");
const NOW = new Date().toISOString();

function summaries(n: number): ConversationSummary[] {
  return Array.from({ length: n }, (_, i) => ({ id: `c${i + 1}`, title: `Conversación ${i + 1}`, updated_at: NOW }));
}

type Route = (url: string, init?: RequestInit) => Response | Promise<Response>;

function routes(handlers: Record<string, Route>) {
  fetchMock.mockImplementation(async (url: string, init?: RequestInit) => {
    const key = `${init?.method ?? "GET"} ${url}`;
    const handler = handlers[key];
    if (!handler) throw new Error(`unexpected fetch ${key}`);
    return handler(url, init);
  });
}

function ActiveProbe() {
  const { state } = useChatContext();
  return <p data-testid="active">{state.conversationId ?? "none"}</p>;
}

function renderList(variant: "nav" | "page") {
  return render(
    <ConversationsProvider>
      <ChatProvider userId="u1">
        <ConversationList variant={variant} />
        <ActiveProbe />
      </ChatProvider>
    </ConversationsProvider>,
  );
}

beforeEach(() => {
  localStorage.clear();
  vi.stubGlobal("fetch", fetchMock);
});

afterEach(() => {
  fetchMock.mockReset();
  router.push.mockReset();
  toast.error.mockReset();
  vi.unstubAllGlobals();
});

describe("ConversationList nav", () => {
  it("lists_recent_in_nav_max_8", async () => {
    routes({ "GET /api/rooster/conversations": () => Response.json(summaries(10)) });
    renderList("nav");
    const items = await screen.findAllByRole("button", { name: /^Conversación \d+$/ });
    expect(items.map((b) => b.textContent)).toEqual(summaries(8).map((c) => c.title));
  });

  it("untitled_shows_placeholder", async () => {
    routes({ "GET /api/rooster/conversations": () => Response.json([{ id: "c1", title: null, updated_at: NOW }]) });
    renderList("nav");
    expect(await screen.findByRole("button", { name: "Nueva conversación" })).toBeInTheDocument();
  });

  it("empty_list_shows_hint", async () => {
    routes({ "GET /api/rooster/conversations": () => Response.json([]) });
    renderList("nav");
    expect(await screen.findByText("Tus conversaciones aparecerán aquí.")).toBeInTheDocument();
  });

  it("nav_click_opens_and_marks_active", async () => {
    routes({
      "GET /api/rooster/conversations": () => Response.json(summaries(2)),
      "GET /api/rooster/conversations/c2": () => Response.json({ id: "c2", title: "Conversación 2", messages: [] }),
    });
    renderList("nav");
    fireEvent.click(await screen.findByRole("button", { name: "Conversación 2" }));

    await waitFor(() => expect(screen.getByTestId("active")).toHaveTextContent("c2"));
    expect(screen.getByRole("button", { name: "Conversación 2" })).toHaveAttribute("aria-current", "true");
    expect(screen.getByRole("button", { name: "Conversación 1" })).not.toHaveAttribute("aria-current");
    expect(localStorage.getItem(KEY)).toBe("c2");
    expect(router.push).not.toHaveBeenCalled();
  });

  it("items_disabled_while_streaming", async () => {
    const body = new ReadableStream<Uint8Array>({ start: () => {} });
    routes({
      "GET /api/rooster/conversations": () => Response.json(summaries(2)),
      "POST /api/rooster/conversations": () => Response.json({ id: "c9" }, { status: 201 }),
      "POST /api/rooster/conversations/c9/messages": () =>
        new Response(body, { status: 200, headers: { "content-type": "text/event-stream" } }),
    });
    function Sender() {
      const { send } = useChatContext();
      return (
        <button type="button" onClick={() => void send("hola")}>
          enviar
        </button>
      );
    }
    render(
      <ConversationsProvider>
        <ChatProvider userId="u1">
          <ConversationList variant="nav" />
          <Sender />
        </ChatProvider>
      </ConversationsProvider>,
    );
    const item = await screen.findByRole("button", { name: "Conversación 1" });
    expect(item).toBeEnabled();
    act(() => screen.getByRole("button", { name: "enviar" }).click());
    await waitFor(() => expect(item).toBeDisabled());
  });
});

describe("ConversationList page", () => {
  it("page_shows_all_with_relative_date_and_navigates", async () => {
    const old = { id: "c11", title: "Antigua", updated_at: new Date(Date.now() - 86_400_000).toISOString() };
    routes({
      "GET /api/rooster/conversations": () => Response.json([...summaries(10), old]),
      "GET /api/rooster/conversations/c11": () => Response.json({ id: "c11", title: "Antigua", messages: [] }),
    });
    renderList("page");
    const row = (await screen.findByRole("button", { name: "Antigua" })).closest("li")!;
    expect(screen.getAllByRole("listitem")).toHaveLength(11);
    expect(within(row).getByText("ayer")).toBeInTheDocument();

    fireEvent.click(within(row).getByRole("button", { name: "Antigua" }));
    await waitFor(() => expect(screen.getByTestId("active")).toHaveTextContent("c11"));
    expect(router.push).toHaveBeenCalledWith("/chat/c11");
  });

  it("rename_inline_enter_saves_esc_cancels", async () => {
    let patched: unknown = null;
    routes({
      "GET /api/rooster/conversations": () => Response.json(summaries(1)),
      "PATCH /api/rooster/conversations/c1": (_u, init) => {
        patched = JSON.parse(String(init!.body));
        return Response.json({ id: "c1", title: "Ruzafa", updated_at: NOW });
      },
    });
    renderList("page");
    await screen.findByRole("button", { name: "Conversación 1" });

    fireEvent.click(screen.getByRole("button", { name: "Renombrar «Conversación 1»" }));
    const input = screen.getByRole("textbox", { name: "Título de la conversación" });
    fireEvent.change(input, { target: { value: "Ignorado" } });
    fireEvent.keyDown(input, { key: "Escape" });
    expect(screen.queryByRole("textbox")).toBeNull();
    expect(screen.getByRole("button", { name: "Conversación 1" })).toBeInTheDocument();
    expect(patched).toBeNull();

    fireEvent.click(screen.getByRole("button", { name: "Renombrar «Conversación 1»" }));
    fireEvent.change(screen.getByRole("textbox"), { target: { value: "  Ruzafa  " } });
    fireEvent.keyDown(screen.getByRole("textbox"), { key: "Enter" });
    expect(await screen.findByRole("button", { name: "Ruzafa" })).toBeInTheDocument();
    expect(patched).toEqual({ title: "Ruzafa" });
  });

  it("rename_optimistic_and_rollback", async () => {
    let fail!: () => void;
    routes({
      "GET /api/rooster/conversations": () => Response.json(summaries(1)),
      "PATCH /api/rooster/conversations/c1": () =>
        new Promise<Response>((r) => (fail = () => r(new Response("", { status: 500 })))),
    });
    renderList("page");
    await screen.findByRole("button", { name: "Conversación 1" });

    fireEvent.click(screen.getByRole("button", { name: "Renombrar «Conversación 1»" }));
    fireEvent.change(screen.getByRole("textbox"), { target: { value: "Ruzafa" } });
    fireEvent.keyDown(screen.getByRole("textbox"), { key: "Enter" });
    // Shown right away, before the server answers.
    expect(await screen.findByRole("button", { name: "Ruzafa" })).toBeInTheDocument();

    await act(async () => fail());
    expect(await screen.findByRole("button", { name: "Conversación 1" })).toBeInTheDocument();
    expect(toast.error).toHaveBeenCalledWith("No se pudo renombrar la conversación. Inténtalo de nuevo.");
  });

  it("delete_active_resets_chat", async () => {
    localStorage.setItem(KEY, "c1");
    let deleted = false;
    routes({
      "GET /api/rooster/conversations": () => Response.json(summaries(2)),
      "GET /api/rooster/conversations/c1": () => Response.json({ id: "c1", title: "Conversación 1", messages: [] }),
      "DELETE /api/rooster/conversations/c1": () => {
        deleted = true;
        return new Response(null, { status: 204 });
      },
    });
    renderList("page");
    await waitFor(() => expect(screen.getByTestId("active")).toHaveTextContent("c1"));

    fireEvent.click(await screen.findByRole("button", { name: "Eliminar «Conversación 1»" }));
    const dialog = screen.getByRole("dialog");
    fireEvent.click(within(dialog).getByRole("button", { name: "Eliminar" }));

    await waitFor(() => expect(screen.queryByRole("button", { name: "Conversación 1" })).toBeNull());
    expect(deleted).toBe(true);
    expect(screen.getByTestId("active")).toHaveTextContent("none");
    expect(localStorage.getItem(KEY)).toBeNull();
    expect(screen.getByRole("button", { name: "Conversación 2" })).toBeInTheDocument();
  });

  it("delete_cancel_keeps_item_and_failure_rolls_back", async () => {
    routes({
      "GET /api/rooster/conversations": () => Response.json(summaries(2)),
      "DELETE /api/rooster/conversations/c2": () => new Response("", { status: 500 }),
    });
    renderList("page");
    fireEvent.click(await screen.findByRole("button", { name: "Eliminar «Conversación 2»" }));
    fireEvent.click(within(screen.getByRole("dialog")).getByRole("button", { name: "Cancelar" }));
    expect(fetchMock.mock.calls.some(([, init]) => init?.method === "DELETE")).toBe(false);

    fireEvent.click(screen.getByRole("button", { name: "Eliminar «Conversación 2»" }));
    fireEvent.click(within(screen.getByRole("dialog")).getByRole("button", { name: "Eliminar" }));
    await waitFor(() => expect(toast.error).toHaveBeenCalledWith("No se pudo eliminar la conversación. Inténtalo de nuevo."));
    // Restored in its original place.
    expect(screen.getAllByRole("button", { name: /^Conversación \d$/ }).map((b) => b.textContent)).toEqual([
      "Conversación 1",
      "Conversación 2",
    ]);
  });
});

describe("ExamplePrompts", () => {
  it("pill_starts_chat_and_shows_it_once_created", async () => {
    localStorage.setItem(KEY, "c1");
    let resolveCreate!: (r: Response) => void;
    routes({
      "GET /api/rooster/conversations": () => Response.json(summaries(1)),
      "GET /api/rooster/conversations/c1": () => Response.json({ id: "c1", title: "x", messages: [] }),
      "POST /api/rooster/conversations": () => new Promise<Response>((r) => (resolveCreate = r)),
      "POST /api/rooster/conversations/new/messages": () =>
        new Response(new ReadableStream({ start: () => {} }), { status: 200, headers: { "content-type": "text/event-stream" } }),
    });
    render(
      <ConversationsProvider>
        <ChatProvider userId="u1">
          <ExamplePrompts items={["¿Dónde invertir?", "Otra"]} />
          <ActiveProbe />
        </ChatProvider>
      </ConversationsProvider>,
    );
    await waitFor(() => expect(screen.getByTestId("active")).toHaveTextContent("c1"));

    fireEvent.click(screen.getByRole("button", { name: "¿Dónde invertir?" }));
    // Reset first; no navigation until the new conversation exists.
    expect(screen.getByTestId("active")).toHaveTextContent("none");
    expect(router.push).not.toHaveBeenCalled();

    await act(async () => resolveCreate(Response.json({ id: "new" }, { status: 201 })));
    await waitFor(() => expect(router.push).toHaveBeenCalledWith("/chat/new"));
    expect(router.push).toHaveBeenCalledTimes(1);
    const post = fetchMock.mock.calls.find(([u]) => u === "/api/rooster/conversations/new/messages");
    expect(JSON.parse(post![1].body)).toEqual({ text: "¿Dónde invertir?" });

    // While that turn streams, the pills do nothing.
    expect(screen.getByRole("button", { name: "Otra" })).toBeDisabled();
  });
});
