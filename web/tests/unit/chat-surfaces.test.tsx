import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from "vitest";
import { ChatSurfaces } from "@/components/chat/ChatSurfaces";
import { FullChat } from "@/components/chat/FullChat";
import { ThemeProvider } from "@/components/theme-provider";
import { ChatProvider, useChatContext } from "@/lib/chat-context";
import { ChatSurfaceProvider } from "@/lib/chat-surface";
import { storageKey } from "@/lib/client";
import type { Preferences, UserOut } from "@/lib/types";
import { ConversationsProvider } from "@/lib/use-conversations";
import { PreferencesProvider } from "@/lib/use-preferences";

const nav = vi.hoisted(() => ({
  pathname: "/",
  search: "",
  router: { push: vi.fn(), replace: vi.fn(), back: vi.fn() },
}));
vi.mock("next/navigation", () => ({
  usePathname: () => nav.pathname,
  useRouter: () => nav.router,
  useSearchParams: () => new URLSearchParams(nav.search),
}));
vi.mock("@/lib/session-actions", () => ({ signOutAction: vi.fn() }));
vi.mock("sonner", () => ({ toast: { error: vi.fn() } }));
vi.mock("@/components/map/ChoroplethMap", () => ({ default: () => <div data-testid="choropleth-mock" /> }));
vi.mock("@/components/map/PointMap", () => ({ default: () => <div data-testid="point-map-mock" /> }));

const USER: UserOut = { id: "u1", email: "maria@example.es", name: "María Fernández", image: null, preferences: {} };
const fetchMock = vi.fn();
const media = vi.hoisted(() => ({ desktop: true }));

beforeAll(() => {
  window.matchMedia = ((query: string) => ({
    // Only the chat's breakpoint query follows `media.desktop`; next-themes asks for the color scheme.
    matches: query.includes("min-width: 1024px") ? media.desktop : false,
    media: query,
    onchange: null,
    addEventListener: () => {},
    removeEventListener: () => {},
    addListener: () => {},
    removeListener: () => {},
    dispatchEvent: () => false,
  })) as typeof window.matchMedia;
  globalThis.ResizeObserver ??= class {
    observe() {}
    unobserve() {}
    disconnect() {}
  } as unknown as typeof ResizeObserver;
  Element.prototype.scrollIntoView ??= () => {};
  Element.prototype.hasPointerCapture ??= () => false;
});

type Route = (url: string, init?: RequestInit) => Response | Promise<Response>;
function routes(handlers: Record<string, Route>) {
  fetchMock.mockImplementation(async (url: string, init?: RequestInit) => {
    const key = `${init?.method ?? "GET"} ${url}`;
    const handler = handlers[key];
    if (!handler) throw new Error(`unexpected fetch ${key}`);
    return handler(url, init);
  });
}

const BASE: Record<string, Route> = {
  "GET /api/rooster/conversations": () => Response.json([]),
  "PATCH /api/rooster/me/preferences": (_u, init) => Response.json(JSON.parse(String(init?.body))),
};

beforeEach(() => {
  nav.pathname = "/";
  nav.search = "";
  media.desktop = true;
  localStorage.clear();
  vi.stubGlobal("fetch", fetchMock);
  routes(BASE);
});
afterEach(() => {
  fetchMock.mockReset();
  nav.router.push.mockReset();
  nav.router.replace.mockReset();
  vi.unstubAllGlobals();
});

function Sender() {
  const { send } = useChatContext();
  return (
    <button type="button" onClick={() => void send("hola")}>
      enviar-prueba
    </button>
  );
}

function Providers({ prefs = {}, children }: { prefs?: Preferences; children: React.ReactNode }) {
  return (
    <ThemeProvider>
      <PreferencesProvider initial={prefs}>
        <ConversationsProvider>
          <ChatProvider userId="u1">
            <ChatSurfaceProvider>{children}</ChatSurfaceProvider>
          </ChatProvider>
        </ConversationsProvider>
      </PreferencesProvider>
    </ThemeProvider>
  );
}

function App({ page }: { page?: React.ReactNode }) {
  return (
    <ChatSurfaces user={USER} conversations={null}>
      <Sender />
      {page}
    </ChatSurfaces>
  );
}

const sidebar = () => screen.queryByRole("complementary", { name: "Chat con Rooster" });
const toggle = () => screen.getByRole("button", { name: "Pregunta a Rooster" });

function patches() {
  return fetchMock.mock.calls
    .filter(([u, init]) => u === "/api/rooster/me/preferences" && init?.method === "PATCH")
    .map(([, init]) => JSON.parse(String(init.body)));
}

describe("ChatSidebar", () => {
  it("sidebar_visible_by_default_and_closes_to_preference", async () => {
    const { unmount } = render(
      <Providers>
        <App />
      </Providers>,
    );
    const aside = sidebar()!;
    expect(aside).toBeInTheDocument();
    expect(within(aside).getByRole("textbox")).toBeInTheDocument();
    expect(toggle()).toHaveAttribute("aria-pressed", "true");

    await act(async () => within(aside).getByRole("button", { name: "Cerrar chat" }).click());
    expect(sidebar()).toBeNull();
    expect(patches()).toEqual([{ chat_open: false }]);
    expect(toggle()).toHaveAttribute("aria-pressed", "false");

    // The top-bar button toggles it back.
    await act(async () => toggle().click());
    expect(sidebar()).toBeInTheDocument();
    expect(patches()).toEqual([{ chat_open: false }, { chat_open: true }]);
    unmount();

    // A stored "closed" preference keeps it closed.
    render(
      <Providers prefs={{ chat_open: false }}>
        <App />
      </Providers>,
    );
    expect(sidebar()).toBeNull();
  });

  it("sidebar_hidden_on_full_screen_route", () => {
    nav.pathname = "/chat/c1";
    routes({ ...BASE, "GET /api/rooster/conversations/c1": () => Response.json({ id: "c1", title: null, messages: [] }) });
    render(
      <Providers>
        <App page={<FullChat id="c1" />} />
      </Providers>,
    );
    expect(sidebar()).toBeNull();
    // No chat toggle (or floating button) while the chat is the page.
    expect(toggle()).toBeDisabled();
    expect(screen.getAllByRole("textbox")).toHaveLength(1);
  });

  it("sidebar_kept_on_chat_list_page", () => {
    nav.pathname = "/chat";
    render(
      <Providers>
        <App />
      </Providers>,
    );
    expect(sidebar()).toBeInTheDocument();
  });

  it("expand_navigates_to_chat_id", async () => {
    localStorage.setItem(storageKey("u1"), "c7");
    routes({ ...BASE, "GET /api/rooster/conversations/c7": () => Response.json({ id: "c7", title: "x", messages: [] }) });
    render(
      <Providers>
        <App />
      </Providers>,
    );
    const expand = within(sidebar()!).getByRole("button", { name: "Abrir a pantalla completa" });
    await waitFor(() => expect(within(sidebar()!).queryByTestId("chat-empty")).toBeInTheDocument());
    await act(async () => expand.click());
    expect(nav.router.push).toHaveBeenCalledWith("/chat/c7");
  });

  it("expand_without_conversation_goes_to_list", async () => {
    render(
      <Providers>
        <App />
      </Providers>,
    );
    await act(async () => within(sidebar()!).getByRole("button", { name: "Abrir a pantalla completa" }).click());
    expect(nav.router.push).toHaveBeenCalledWith("/chat");
  });
});

describe("Chat surfaces share one live turn", () => {
  it("same_live_turn_shown_after_switching_surface", async () => {
    const enc = new TextEncoder();
    let stream!: ReadableStreamDefaultController<Uint8Array>;
    const body = new ReadableStream<Uint8Array>({ start: (c) => void (stream = c) });
    const emit = (e: object) =>
      stream.enqueue(enc.encode(`event: ${(e as { type: string }).type}\ndata: ${JSON.stringify(e)}\n\n`));
    routes({
      ...BASE,
      "POST /api/rooster/conversations": () => Response.json({ id: "c1" }, { status: 201 }),
      "POST /api/rooster/conversations/c1/messages": () =>
        new Response(body, { status: 200, headers: { "content-type": "text/event-stream" } }),
      "GET /api/rooster/conversations/c1": () => Response.json({ id: "c1", title: null, messages: [] }),
    });

    const { rerender } = render(
      <Providers>
        <App />
      </Providers>,
    );
    await act(async () => screen.getByRole("button", { name: "enviar-prueba" }).click());
    await act(async () => emit({ type: "text_delta", text: "Benimaclet rinde más" }));
    expect(within(sidebar()!).getByText("Benimaclet rinde más")).toBeInTheDocument();

    // Expand: the full-screen route renders the same provider state.
    nav.pathname = "/chat/c1";
    rerender(
      <Providers>
        <App page={<FullChat id="c1" />} />
      </Providers>,
    );
    expect(sidebar()).toBeNull();
    expect(screen.getByText("Benimaclet rinde más")).toBeInTheDocument();
    await act(async () => emit({ type: "text_delta", text: " que Russafa" }));
    expect(screen.getByText("Benimaclet rinde más que Russafa")).toBeInTheDocument();

    const posts = fetchMock.mock.calls.filter(([, init]) => init?.method === "POST");
    expect(posts).toHaveLength(2); // create + one message; no new request on switching
    await act(async () => {
      emit({ type: "done" });
      stream.close();
    });
    nav.pathname = "/";
  });
});

describe("Composer draft", () => {
  it("draft_survives_surface_switches", async () => {
    const { rerender } = render(
      <Providers>
        <App />
      </Providers>,
    );
    fireEvent.change(within(sidebar()!).getByRole("textbox"), {
      target: { value: "borrador" },
    });

    // Sidebar -> full screen.
    nav.pathname = "/chat/c1";
    routes({
      ...BASE,
      "GET /api/rooster/conversations/c1": () => Response.json({ id: "c1", title: null, messages: [] }),
    });
    rerender(
      <Providers>
        <App page={<FullChat id="c1" />} />
      </Providers>,
    );
    expect(screen.getByRole("textbox")).toHaveValue("borrador");

    // Full screen -> phone sheet.
    nav.pathname = "/";
    media.desktop = false;
    rerender(
      <Providers>
        <App />
      </Providers>,
    );
    await act(async () => screen.getByRole("button", { name: "Abrir chat" }).click());
    expect(await screen.findByRole("textbox")).toHaveValue("borrador");
  });
});

describe("Full screen route navigation", () => {
  it("new_conversation_on_full_route_replaces_url_with_chat", async () => {
    nav.pathname = "/chat/c1";
    routes({
      ...BASE,
      "GET /api/rooster/conversations/c1": () => Response.json({ id: "c1", title: null, messages: [] }),
    });
    render(
      <Providers>
        <App page={<FullChat id="c1" />} />
      </Providers>,
    );
    await waitFor(() => expect(screen.getByRole("button", { name: /Nueva conversación/ })).toBeEnabled());
    nav.router.replace.mockClear();
    await act(async () => screen.getByRole("button", { name: /Nueva conversación/ }).click());
    expect(nav.router.replace).toHaveBeenCalledWith("/chat");
  });

  it("back_keeps_previous_query_string", async () => {
    nav.pathname = "/barrios";
    nav.search = "orden=yield&q=a";
    routes({
      ...BASE,
      "GET /api/rooster/conversations/c1": () => Response.json({ id: "c1", title: null, messages: [] }),
    });
    const { rerender } = render(
      <Providers>
        <App />
      </Providers>,
    );
    nav.pathname = "/chat/c1";
    nav.search = "";
    rerender(
      <Providers>
        <App page={<FullChat id="c1" />} />
      </Providers>,
    );
    await act(async () => screen.getByRole("button", { name: "Volver" }).click());
    expect(nav.router.push).toHaveBeenCalledWith("/barrios?orden=yield&q=a");
  });
});

describe("ChatSheet", () => {
  it("sheet_used_below_1024", async () => {
    media.desktop = false;
    render(
      <Providers>
        <App />
      </Providers>,
    );
    expect(sidebar()).toBeNull();
    expect(screen.queryByRole("dialog")).toBeNull();
    const fab = screen.getByRole("button", { name: "Abrir chat" });
    // Browsers focus a clicked button; jsdom does not.
    fab.focus();
    await act(async () => fab.click());
    const sheet = screen.getByRole("dialog", { name: "Chat con Rooster" });
    // Focus moves into the sheet (the sheet itself, so the phone keyboard does not pop up).
    expect(sheet).toHaveFocus();
    // Closing it restores focus to what opened it.
    await act(async () => within(sheet).getByRole("button", { name: "Cerrar chat" }).click());
    expect(screen.queryByRole("dialog")).toBeNull();
    // Radix restores focus on a timer after unmount.
    await waitFor(() => expect(fab).toHaveFocus());

    // The top-bar button opens it too and gets focus back.
    toggle().focus();
    await act(async () => toggle().click());
    expect(screen.getByRole("dialog", { name: "Chat con Rooster" })).toBeInTheDocument();
    expect(patches()).toEqual([]);
    await act(async () => fireEvent.keyDown(document.activeElement!, { key: "Escape" }));
    expect(screen.queryByRole("dialog")).toBeNull();
    await waitFor(() => expect(toggle()).toHaveFocus());
  });

  it("sheet_expand_goes_full_screen", async () => {
    media.desktop = false;
    render(
      <Providers>
        <App />
      </Providers>,
    );
    await act(async () => screen.getByRole("button", { name: "Abrir chat" }).click());
    const sheet = screen.getByRole("dialog", { name: "Chat con Rooster" });
    await act(async () => within(sheet).getByRole("button", { name: "Abrir a pantalla completa" }).click());
    expect(nav.router.push).toHaveBeenCalledWith("/chat");
    expect(screen.queryByRole("dialog")).toBeNull();
  });
});

describe("FullChat", () => {
  it("full_route_404_redirects_to_chat", async () => {
    nav.pathname = "/chat/gone";
    routes({ ...BASE, "GET /api/rooster/conversations/gone": () => Response.json({ detail: "x" }, { status: 404 }) });
    render(
      <Providers>
        <App page={<FullChat id="gone" />} />
      </Providers>,
    );
    await waitFor(() => expect(nav.router.replace).toHaveBeenCalledWith("/chat"));
  });

  it("full_route_opens_conversation_and_back_returns_with_sidebar_open", async () => {
    nav.pathname = "/";
    routes({
      ...BASE,
      "GET /api/rooster/conversations/c3": () =>
        Response.json({ id: "c3", title: "Benimaclet", messages: [{ role: "user", content: "pregunta guardada" }] }),
    });
    const { rerender } = render(
      <Providers prefs={{ chat_open: false }}>
        <App />
      </Providers>,
    );
    nav.pathname = "/chat/c3";
    rerender(
      <Providers prefs={{ chat_open: false }}>
        <App page={<FullChat id="c3" />} />
      </Providers>,
    );
    expect(await screen.findByText("pregunta guardada")).toBeInTheDocument();
    await act(async () => screen.getByRole("button", { name: "Volver" }).click());
    expect(nav.router.push).toHaveBeenCalledWith("/");
    expect(patches()).toEqual([{ chat_open: true }]);
    expect(nav.router.replace).not.toHaveBeenCalled();
  });
});
