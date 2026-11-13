import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ConversationList } from "@/components/chat/ConversationList";
import { ExamplePrompts } from "@/components/chat/ExamplePrompts";
import { ChatProvider as RealChatProvider, useChatContext } from "@/lib/chat-context";
import { ChatSurfaceProvider, useChatSurface } from "@/lib/chat-surface";
import { storageKey } from "@/lib/client";
import { ConversationsProvider, useConversations } from "@/lib/use-conversations";
import { PreferencesProvider } from "@/lib/use-preferences";
import type { ConversationSummary, Preferences } from "@/lib/types";

/** The chat plus where it is shown (sidebar/sheet), as in the (app) layout. */
const prefsHolder: { initial: Preferences } = { initial: {} };
function ChatProvider({ userId, children }: { userId: string; children: React.ReactNode }) {
  return (
    <PreferencesProvider initial={prefsHolder.initial}>
      <RealChatProvider userId={userId}>
        <ChatSurfaceProvider>
          {children}
          <SurfaceProbe />
        </ChatSurfaceProvider>
      </RealChatProvider>
    </PreferencesProvider>
  );
}

function SurfaceProbe() {
  const { sidebarVisible } = useChatSurface();
  return <p data-testid="sidebar">{sidebarVisible ? "open" : "closed"}</p>;
}

const router = vi.hoisted(() => ({ push: vi.fn() }));
vi.mock("next/navigation", () => ({ useRouter: () => router, usePathname: () => "/" }));
const toast = vi.hoisted(() => ({ error: vi.fn() }));
vi.mock("sonner", () => ({ toast }));

const fetchMock = vi.fn();
const KEY = storageKey("u1");
const NOW = new Date().toISOString();

function summaries(n: number): ConversationSummary[] {
  return Array.from({ length: n }, (_, i) => ({ id: `c${i + 1}`, title: `Conversación ${i + 1}`, updated_at: NOW }));
}

/** Streams a test leaves open; closed after each test. */
const openStreams: ReadableStreamDefaultController<Uint8Array>[] = [];
function sseStream(): { response: () => Response; emit: (e: object) => void; close: () => void } {
  const enc = new TextEncoder();
  let ctrl!: ReadableStreamDefaultController<Uint8Array>;
  const body = new ReadableStream<Uint8Array>({ start: (c) => void (ctrl = c) });
  openStreams.push(ctrl);
  return {
    response: () => new Response(body, { status: 200, headers: { "content-type": "text/event-stream" } }),
    emit: (e) => ctrl.enqueue(enc.encode(`event: ${(e as { type: string }).type}\ndata: ${JSON.stringify(e)}\n\n`)),
    close: () => ctrl.close(),
  };
}

function Sender() {
  const { send } = useChatContext();
  return (
    <button type="button" onClick={() => void send("hola")}>
      enviar
    </button>
  );
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

function renderList(variant: "nav" | "recent" | "page") {
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
  prefsHolder.initial = {};
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
    const { unmount } = renderList("nav");
    expect(await screen.findByText("Sin conversaciones todavía")).toBeInTheDocument();
    unmount();
    renderList("recent");
    expect(await screen.findByText("Tus conversaciones aparecerán aquí.")).toBeInTheDocument();
  });

  it("recent_click_opens_in_sidebar_without_navigating", async () => {
    prefsHolder.initial = { chat_open: false };
    routes({
      "GET /api/rooster/conversations": () => Response.json(summaries(10)),
      "GET /api/rooster/conversations/c2": () => Response.json({ id: "c2", title: "Conversación 2", messages: [] }),
      "PATCH /api/rooster/me/preferences": () => Response.json({ chat_open: true }),
    });
    renderList("recent");
    const items = await screen.findAllByRole("button", { name: /^Conversación \d+$/ });
    expect(items).toHaveLength(8);
    expect(screen.getByTestId("sidebar")).toHaveTextContent("closed");
    fireEvent.click(screen.getByRole("button", { name: "Conversación 2" }));
    await waitFor(() => expect(screen.getByTestId("active")).toHaveTextContent("c2"));
    expect(screen.getByTestId("sidebar")).toHaveTextContent("open");
    expect(router.push).not.toHaveBeenCalled();
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
    const stream = sseStream();
    routes({
      "GET /api/rooster/conversations": () => Response.json(summaries(2)),
      "POST /api/rooster/conversations": () => Response.json({ id: "c9" }, { status: 201 }),
      "POST /api/rooster/conversations/c9/messages": stream.response,
    });
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
  it("rename_commits_on_blur", async () => {
    let patched: unknown = null;
    routes({
      "GET /api/rooster/conversations": () => Response.json(summaries(1)),
      "PATCH /api/rooster/conversations/c1": (_u, init) => {
        patched = JSON.parse(String(init!.body));
        return Response.json({ id: "c1", title: "Benimaclet", updated_at: NOW });
      },
    });
    renderList("page");
    fireEvent.click(await screen.findByRole("button", { name: "Renombrar «Conversación 1»" }));
    fireEvent.change(screen.getByRole("textbox"), { target: { value: "Benimaclet" } });
    fireEvent.blur(screen.getByRole("textbox"));
    expect(await screen.findByRole("button", { name: "Benimaclet" })).toBeInTheDocument();
    expect(patched).toEqual({ title: "Benimaclet" });
  });

  it("rename_and_delete_disabled_for_busy_conversation_including_refetch", async () => {
    localStorage.setItem(KEY, "c1");
    const stream = sseStream();
    let gets = 0;
    let resolveRefetch!: (r: Response) => void;
    routes({
      "GET /api/rooster/conversations": () => Response.json(summaries(2)),
      "GET /api/rooster/conversations/c1": () =>
        gets++ === 0
          ? Response.json({ id: "c1", title: "x", messages: [] })
          : new Promise<Response>((r) => (resolveRefetch = r)),
      "POST /api/rooster/conversations/c1/messages": stream.response,
    });
    render(
      <ConversationsProvider>
        <ChatProvider userId="u1">
          <ConversationList variant="page" />
          <Sender />
          <ActiveProbe />
        </ChatProvider>
      </ConversationsProvider>,
    );
    await waitFor(() => expect(screen.getByTestId("active")).toHaveTextContent("c1"));
    const renameActive = await screen.findByRole("button", { name: "Renombrar «Conversación 1»" });
    const deleteActive = screen.getByRole("button", { name: "Eliminar «Conversación 1»" });
    expect(deleteActive).toBeEnabled();

    act(() => screen.getByRole("button", { name: "enviar" }).click());
    await waitFor(() => expect(deleteActive).toBeDisabled());
    expect(renameActive).toBeDisabled();
    // Other conversations can still be renamed and deleted.
    expect(screen.getByRole("button", { name: "Eliminar «Conversación 2»" })).toBeEnabled();
    expect(screen.getByRole("button", { name: "Renombrar «Conversación 2»" })).toBeEnabled();

    // After `done`, while the stored turn is refetched, still disabled.
    await act(async () => {
      stream.emit({ type: "done" });
      stream.close();
    });
    await waitFor(() => expect(gets).toBe(2));
    expect(deleteActive).toBeDisabled();
    expect(screen.getByRole("button", { name: "Conversación 2" })).toBeDisabled();

    await act(async () => resolveRefetch(Response.json({ id: "c1", title: "x", messages: [] })));
    await waitFor(() => expect(deleteActive).toBeEnabled());
    expect(renameActive).toBeEnabled();
  });

  it("rename_resorts_by_updated_at", async () => {
    const older = new Date(Date.now() - 3_600_000).toISOString();
    routes({
      "GET /api/rooster/conversations": () =>
        Response.json([
          { id: "c1", title: "Primera", updated_at: NOW },
          { id: "c2", title: "Segunda", updated_at: older },
        ]),
      "PATCH /api/rooster/conversations/c2": () =>
        Response.json({ id: "c2", title: "Renombrada", updated_at: new Date(Date.now() + 1000).toISOString() }),
    });
    renderList("page");
    fireEvent.click(await screen.findByRole("button", { name: "Renombrar «Segunda»" }));
    fireEvent.change(screen.getByRole("textbox"), { target: { value: "Renombrada" } });
    fireEvent.keyDown(screen.getByRole("textbox"), { key: "Enter" });
    await waitFor(() =>
      expect(screen.getAllByRole("button", { name: /^(Primera|Renombrada)$/ }).map((b) => b.textContent)).toEqual([
        "Renombrada",
        "Primera",
      ]),
    );
  });

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
  it("pill_starts_chat_and_shows_it_in_the_sidebar", async () => {
    prefsHolder.initial = { chat_open: false };
    localStorage.setItem(KEY, "c1");
    let resolveCreate!: (r: Response) => void;
    routes({
      "GET /api/rooster/conversations": () => Response.json(summaries(1)),
      "GET /api/rooster/conversations/c1": () => Response.json({ id: "c1", title: "x", messages: [] }),
      "POST /api/rooster/conversations": () => new Promise<Response>((r) => (resolveCreate = r)),
      "POST /api/rooster/conversations/new/messages": sseStream().response,
      "PATCH /api/rooster/me/preferences": () => Response.json({ chat_open: true }),
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
    // A new conversation starts and the sidebar opens at once; the page stays where it is.
    expect(screen.getByTestId("active")).toHaveTextContent("none");
    expect(screen.getByTestId("sidebar")).toHaveTextContent("open");

    await act(async () => resolveCreate(Response.json({ id: "new" }, { status: 201 })));
    await waitFor(() => expect(screen.getByTestId("active")).toHaveTextContent("new"));
    expect(router.push).not.toHaveBeenCalled();
    const post = fetchMock.mock.calls.find(([u]) => u === "/api/rooster/conversations/new/messages");
    expect(JSON.parse(post![1].body)).toEqual({ text: "¿Dónde invertir?" });

    // Pills wrap as rounded boxes on phones, full pills from sm up.
    expect(screen.getByRole("button", { name: "Otra" }).className).toMatch(/\brounded-2xl\b.*\bsm:rounded-full\b/);
    // While that turn streams, the pills do nothing.
    expect(screen.getByRole("button", { name: "Otra" })).toBeDisabled();
  });
});

describe("useConversations", () => {
  it("slow_initial_load_does_not_overwrite_newer_refresh", async () => {
    let resolveFirst!: (r: Response) => void;
    let lists = 0;
    routes({
      "GET /api/rooster/conversations": () =>
        lists++ === 0 ? new Promise<Response>((r) => (resolveFirst = r)) : Response.json(summaries(2)),
    });
    function Refresher() {
      const { items, refresh } = useConversations();
      return (
        <>
          <button type="button" onClick={() => void refresh()}>
            refrescar
          </button>
          <p data-testid="count">{items.length}</p>
        </>
      );
    }
    render(
      <ConversationsProvider>
        <Refresher />
      </ConversationsProvider>,
    );
    await act(async () => screen.getByRole("button", { name: "refrescar" }).click());
    await waitFor(() => expect(screen.getByTestId("count")).toHaveTextContent("2"));
    await act(async () => resolveFirst(Response.json(summaries(1))));
    expect(screen.getByTestId("count")).toHaveTextContent("2");
  });
});
