import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { readFileSync } from "node:fs";
import path from "node:path";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { ChatState, LiveTurn } from "@/lib/chat-state";
import type { AssistantMessage, Message, PrimitivePayload } from "@/lib/types";

vi.mock("@/components/map/ChoroplethMap", () => ({ default: () => <div data-testid="choropleth-mock" /> }));
vi.mock("@/components/map/PointMap", () => ({ default: () => <div data-testid="point-map-mock" /> }));
vi.mock("@/components/primitives/PlotlyFigure", () => ({ default: () => <div data-testid="plotly-mock" /> }));

const fixture = JSON.parse(
  readFileSync(path.resolve(__dirname, "../../../tests/fixtures/primitive_payloads.json"), "utf8"),
) as { payloads: PrimitivePayload[]; folium_code: PrimitivePayload };
const ALL_KINDS = [...fixture.payloads, fixture.folium_code];

const chat = vi.hoisted(() => ({
  state: {} as ChatState,
  busy: false,
  send: vi.fn<(text: string) => Promise<void>>(async () => {}),
  reset: vi.fn(),
}));
vi.mock("@/lib/chat-context", () => ({ useChatContext: () => chat }));

import { ChatPanel } from "@/components/chat/ChatPanel";

const ChatView = () => <ChatPanel density="full" />;

function assistant(over: Partial<AssistantMessage> = {}): AssistantMessage {
  return {
    role: "assistant",
    blocks: [],
    summary: "",
    primitive_payloads: [],
    follow_ups: [],
    graceful_fallback: null,
    ...over,
  };
}

function live(over: Partial<LiveTurn> = {}): LiveTurn {
  return {
    userText: "¿Dónde invierto?",
    thoughts: [],
    text: "",
    primitives: [],
    followUps: [],
    error: null,
    status: "streaming",
    ...over,
  };
}

function setState(over: Partial<ChatState>, busy = false) {
  chat.busy = busy;
  chat.state = { conversationId: "c1", messages: [], live: null, notice: null, draft: null, loading: false, ...over };
}

/**
 * The message list scrolls inside its own container. jsdom has no layout, so scroll geometry is
 * faked on HTMLElement.prototype; `writes` records every programmatic scrollTop assignment.
 */
function fakeScroll(scrollHeight = 5000, clientHeight = 500) {
  const tops = new WeakMap<Element, number>();
  const writes: number[] = [];
  Object.defineProperty(HTMLElement.prototype, "scrollTop", {
    configurable: true,
    get(this: Element) {
      return tops.get(this) ?? 0;
    },
    set(this: Element, v: number) {
      writes.push(v);
      tops.set(this, v);
    },
  });
  Object.defineProperty(HTMLElement.prototype, "scrollHeight", { configurable: true, get: () => scrollHeight });
  Object.defineProperty(HTMLElement.prototype, "clientHeight", { configurable: true, get: () => clientHeight });
  const scroller = () => screen.getByTestId("chat-scroll");
  return {
    writes,
    /** The reader scrolls the list to `y`. */
    readerScrollTo(y: number) {
      tops.set(scroller(), y);
      fireEvent.scroll(scroller());
    },
    restore() {
      for (const k of ["scrollTop", "scrollHeight", "clientHeight"])
        delete (HTMLElement.prototype as unknown as Record<string, unknown>)[k];
    },
  };
}

const input = () => screen.getByPlaceholderText("Pregunta por anuncios, rentabilidades, barrios…");

beforeEach(() => {
  chat.send.mockClear();
  chat.reset.mockClear();
  setState({});
});

describe("ChatPanel (full density, v3 behaviors)", () => {
  it("renders_history_blocks_in_order", () => {
    const messages: Message[] = [
      { role: "user", content: "Pregunta uno" },
      assistant({
        blocks: [
          { kind: "text", content: "Bloque **primero**", role: "lead" },
          { kind: "kpi", label: "Bloque segundo", value: "5,9", unit: "%", delta: null },
          { kind: "text", content: "Bloque tercero", role: "analysis" },
        ],
      }),
    ];
    setState({ messages });
    const { container } = render(<ChatView />);
    const text = container.textContent ?? "";
    const order = ["Pregunta uno", "primero", "Bloque segundo", "Bloque tercero"].map((s) => text.indexOf(s));
    expect(order.every((i) => i >= 0)).toBe(true);
    expect([...order].sort((a, b) => a - b)).toEqual(order);
    expect(container.querySelector("strong")).toHaveTextContent("primero");
  });

  it("latest_follow_ups_clickable_older_plain", () => {
    setState({
      messages: [
        { role: "user", content: "a" },
        assistant({ blocks: [{ kind: "text", content: "uno", role: "analysis" }], follow_ups: ["Pregunta vieja"] }),
        { role: "user", content: "b" },
        assistant({ blocks: [{ kind: "text", content: "dos", role: "analysis" }], follow_ups: ["Pregunta nueva"] }),
      ],
    });
    render(<ChatView />);
    expect(screen.getByRole("button", { name: "Pregunta nueva" })).toBeEnabled();
    expect(screen.queryByRole("button", { name: "Pregunta vieja" })).toBeNull();
    expect(screen.getByText(/Pregunta vieja/)).toBeInTheDocument();
  });

  it("live_turn_owns_pills_over_history", () => {
    setState({
      messages: [{ role: "user", content: "a" }, assistant({ follow_ups: ["Pregunta vieja"] })],
      live: live({ text: "hecho", followUps: ["Pregunta viva"], error: { reason: "r", suggestion: "s" }, status: "done" }),
    });
    render(<ChatView />);
    expect(screen.getByRole("button", { name: "Pregunta viva" })).toBeEnabled();
    expect(screen.queryByRole("button", { name: "Pregunta vieja" })).toBeNull();
  });

  it("follow_up_click_sends_text", () => {
    setState({ messages: [{ role: "user", content: "a" }, assistant({ follow_ups: ["¿Y en Benimaclet?"] })] });
    render(<ChatView />);
    fireEvent.click(screen.getByRole("button", { name: "¿Y en Benimaclet?" }));
    expect(chat.send).toHaveBeenCalledWith("¿Y en Benimaclet?");
  });

  it("fallback_card_shows_reason_and_suggestion", () => {
    setState({
      messages: [
        { role: "user", content: "a" },
        assistant({
          summary: "no debería verse",
          graceful_fallback: { reason: "No encontré datos.", suggestion: "Prueba con otro barrio." },
        }),
      ],
      live: live({ error: { reason: "Se cortó.", suggestion: "Reintenta." }, status: "done" }),
    });
    render(<ChatView />);
    expect(screen.getByText("No encontré datos.")).toBeInTheDocument();
    expect(screen.getByText("Prueba con otro barrio.")).toBeInTheDocument();
    expect(screen.queryByText("no debería verse")).toBeNull();
    expect(screen.getByText("Se cortó.")).toBeInTheDocument();
    expect(screen.getByText("Reintenta.")).toBeInTheDocument();
  });

  it("legacy_message_renders_summary_and_payloads", () => {
    setState({
      messages: [
        { role: "user", content: "a" },
        assistant({
          summary: "Resumen **antiguo**",
          primitive_payloads: [{ kind: "kpi", label: "KPI antiguo", value: "7", unit: null, delta: null }],
        }),
      ],
    });
    const { container } = render(<ChatView />);
    const text = container.textContent ?? "";
    expect(container.querySelector("strong")).toHaveTextContent("antiguo");
    expect(text.indexOf("Resumen")).toBeLessThan(text.indexOf("KPI antiguo"));
  });

  it("composer_disabled_while_busy_or_loading", () => {
    // The hook's `busy` flag covers the stream and the stored refetch after it.
    setState({ live: live() }, true);
    const { rerender } = render(<ChatView />);
    expect(input()).toBeDisabled();
    expect(screen.getByRole("button", { name: "Enviar" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Nueva conversación" })).toBeDisabled();

    setState({ live: live({ status: "done" }) }, true);
    rerender(<ChatView />);
    expect(input()).toBeDisabled();

    setState({ loading: true, conversationId: null });
    rerender(<ChatView />);
    expect(input()).toBeDisabled();

    setState({ live: live({ status: "done", error: { reason: "r", suggestion: "s" } }) });
    rerender(<ChatView />);
    expect(input()).toBeEnabled();
    expect(screen.getByRole("button", { name: "Nueva conversación" })).toBeEnabled();
  });

  it("ime_composition_enter_does_not_send", () => {
    render(<ChatView />);
    fireEvent.change(input(), { target: { value: "hola" } });
    fireEvent.keyDown(input(), { key: "Enter", isComposing: true });
    expect(chat.send).not.toHaveBeenCalled();
  });

  it("enter_sends_shift_enter_newline", () => {
    render(<ChatView />);
    fireEvent.change(input(), { target: { value: "  ¿Qué barrio rinde más?  " } });
    // Shift+Enter keeps the default (newline) and does not send.
    expect(fireEvent.keyDown(input(), { key: "Enter", shiftKey: true })).toBe(true);
    expect(chat.send).not.toHaveBeenCalled();
    expect(fireEvent.keyDown(input(), { key: "Enter" })).toBe(false);
    expect(chat.send).toHaveBeenCalledWith("¿Qué barrio rinde más?");
    expect(input()).toHaveValue("");
  });

  it("blank_input_does_not_send", () => {
    render(<ChatView />);
    fireEvent.change(input(), { target: { value: "   " } });
    fireEvent.keyDown(input(), { key: "Enter" });
    expect(chat.send).not.toHaveBeenCalled();
  });

  it("counter_appears_after_1800_chars", () => {
    render(<ChatView />);
    expect(input()).toHaveAttribute("maxLength", "2000");
    fireEvent.change(input(), { target: { value: "a".repeat(1800) } });
    expect(screen.queryByText("1800/2000")).toBeNull();
    fireEvent.change(input(), { target: { value: "a".repeat(1801) } });
    expect(screen.getByText("1801/2000")).toBeInTheDocument();
  });

  it("new_conversation_calls_reset", () => {
    render(<ChatView />);
    fireEvent.click(screen.getByRole("button", { name: "Nueva conversación" }));
    expect(chat.reset).toHaveBeenCalled();
  });

  it("busy_notice_visible", () => {
    setState({ notice: "Rooster sigue respondiendo a tu mensaje anterior." });
    render(<ChatView />);
    expect(screen.getByRole("status")).toHaveTextContent("Rooster sigue respondiendo a tu mensaje anterior.");
  });

  it("refused_send_text_restored_to_composer", () => {
    const { rerender } = render(<ChatView />);
    fireEvent.change(input(), { target: { value: "otra pregunta" } });
    fireEvent.click(screen.getByRole("button", { name: "Enviar" }));
    expect(input()).toHaveValue("");

    setState({ notice: "Rooster sigue respondiendo a tu mensaje anterior.", draft: { text: "otra pregunta" } });
    rerender(<ChatView />);
    expect(input()).toHaveValue("otra pregunta");
  });

  it("thinking_box_collapses_on_text", () => {
    setState({ live: live({ thoughts: ["Buscando barrios", "Calculando rentabilidad"] }) });
    const { rerender } = render(<ChatView />);
    const toggle = screen.getByRole("button", { name: /Pensando…/ });
    expect(toggle).toHaveTextContent("Calculando rentabilidad");
    expect(toggle).toHaveAttribute("aria-expanded", "true");
    expect(screen.getByText("Buscando barrios")).toBeInTheDocument();

    setState({ live: live({ thoughts: ["Buscando barrios", "Calculando rentabilidad"], text: "Benimaclet" }) });
    rerender(<ChatView />);
    expect(screen.getByRole("button", { name: /Pensando…/ })).toHaveAttribute("aria-expanded", "false");
    expect(screen.queryByText("Buscando barrios")).toBeNull();
    expect(screen.getByText("Benimaclet")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /Pensando…/ }));
    expect(screen.getByText("Buscando barrios")).toBeInTheDocument();
  });

  it("thinking_box_reopens_for_next_turn_after_failed_turn", () => {
    setState({
      live: live({ thoughts: ["a"], text: "parcial", error: { reason: "r", suggestion: "s" }, status: "done" }),
    });
    const { rerender } = render(<ChatView />);
    expect(screen.getByRole("button", { name: /Pensando…/ })).toHaveAttribute("aria-expanded", "false");
    setState({ live: live({ userText: "otra", thoughts: ["b"] }) });
    rerender(<ChatView />);
    expect(screen.getByRole("button", { name: /Pensando…/ })).toHaveAttribute("aria-expanded", "true");
  });

  it("duplicate_follow_ups_render_each", () => {
    setState({ messages: [{ role: "user", content: "a" }, assistant({ follow_ups: ["Igual", "Igual"] })] });
    render(<ChatView />);
    expect(screen.getAllByRole("button", { name: "Igual" })).toHaveLength(2);
  });

  it("auto_scroll_pauses_when_scrolled_up_and_resumes_on_new_turn", () => {
    const scroll = fakeScroll();
    try {
      setState({ messages: [{ role: "user", content: "a" }, assistant({ follow_ups: [] })] });
      const { rerender } = render(<ChatView />);
      expect(scroll.writes).toContain(5000);
      scroll.readerScrollTo(3000);
      scroll.readerScrollTo(1000); // scrolled up, far from the bottom
      scroll.writes.length = 0;
      setState({ messages: [{ role: "user", content: "a" }, assistant({ summary: "más" })] });
      rerender(<ChatView />);
      expect(scroll.writes).toEqual([]);
      setState({ messages: chat.state.messages, live: live() });
      rerender(<ChatView />);
      expect(scroll.writes).toContain(5000);
    } finally {
      scroll.restore();
    }
  });

  it("growing_content_keeps_following_only_while_at_bottom", () => {
    const scroll = fakeScroll();
    const observers: (() => void)[] = [];
    vi.stubGlobal(
      "ResizeObserver",
      class {
        constructor(cb: () => void) {
          observers.push(cb);
        }
        observe() {}
        disconnect() {}
      },
    );
    const grow = () => observers.forEach((cb) => cb());
    try {
      setState({ messages: [{ role: "user", content: "a" }, assistant({ follow_ups: [] })] });
      render(<ChatView />);
      expect(observers.length).toBeGreaterThan(0);
      // A map or plot finishing its layout grows the list: the reader at the bottom stays there.
      scroll.writes.length = 0;
      grow();
      expect(scroll.writes).toContain(5000);

      // Our own scroll can land short of a bottom that grew meanwhile: still following.
      scroll.readerScrollTo(3000);
      scroll.writes.length = 0;
      grow();
      expect(scroll.writes).toContain(5000);

      scroll.readerScrollTo(1000); // the reader scrolled up
      scroll.writes.length = 0;
      grow();
      expect(scroll.writes).toEqual([]);
    } finally {
      vi.unstubAllGlobals();
      scroll.restore();
    }
  });

  it("live_turn_shows_user_text_and_primitives", () => {
    setState({ live: live({ text: "Texto vivo", primitives: [{ kind: "kpi", label: "KPI vivo", value: "1", unit: null, delta: null }] }) });
    render(<ChatView />);
    expect(screen.getByText("¿Dónde invierto?")).toBeInTheDocument();
    expect(screen.getByText("Texto vivo")).toBeInTheDocument();
    expect(screen.getByText("KPI vivo")).toBeInTheDocument();
  });

  it("empty_state_examples_send", () => {
    render(<ChatView />);
    const empty = screen.getByTestId("chat-empty");
    expect(within(empty).getByText("Rooster")).toBeInTheDocument();
    expect(within(empty).getByText("Tu copiloto inmobiliario para Valencia")).toBeInTheDocument();
    const examples = within(empty).getAllByRole("button");
    expect(examples.length).toBeGreaterThanOrEqual(3);
    expect(examples.length).toBeLessThanOrEqual(4);
    fireEvent.click(examples[0]);
    expect(chat.send).toHaveBeenCalledWith(examples[0].textContent);
  });

  it("no_empty_state_while_loading", () => {
    setState({ loading: true, conversationId: null });
    render(<ChatView />);
    expect(screen.queryByTestId("chat-empty")).toBeNull();
  });
});

function table(cols: number, rows: number): PrimitivePayload {
  const fields = Array.from({ length: cols }, (_, i) => `c${i}`);
  return {
    kind: "table",
    rows_json: JSON.stringify(Array.from({ length: rows }, (_, r) => Object.fromEntries(fields.map((f) => [f, `${f}-${r}`])))),
    columns: fields.map((f) => ({ field: f, header: `Col ${f}`, formatter: "text", alignment: "left", width_hint: "medium" })),
  };
}

describe("ChatPanel (compact density)", () => {
  it("compact_inlines_small_table_and_cards_big_blocks", async () => {
    setState({
      messages: [
        { role: "user", content: "todo" },
        assistant({ blocks: [...ALL_KINDS, table(4, 6), table(5, 2), table(2, 7)] }),
      ],
    });
    render(<ChatPanel density="compact" />);
    const blocks = screen.getAllByTestId("message-block");
    expect(blocks).toHaveLength(ALL_KINDS.length + 3);
    const cards = screen.getAllByRole("button", { name: /Ampliar/ });
    const names = cards.map((c) => c.textContent);
    // text, kpi, the fixture's 2x2 table and the 4x6 table stay inline; everything else is a card.
    expect(screen.getByText("Benimaclet lidera la rentabilidad bruta de la ciudad.")).toBeInTheDocument();
    expect(screen.getAllByText("Rentabilidad media en Valencia").length).toBeGreaterThan(0);
    expect(screen.getAllByRole("table")).toHaveLength(2);
    expect(screen.getByRole("columnheader", { name: /Col c3/ })).toBeInTheDocument();
    expect(cards).toHaveLength(10);
    // Own title/heading/label first; the Spanish kind label otherwise.
    expect(names).toEqual(
      expect.arrayContaining([
        expect.stringContaining("Rentabilidad bruta por barrio (%)"), // plotly layout title
        expect.stringContaining("Gráfico personalizado"), // folium (no title)
        expect.stringContaining("Mapa de barrios"),
        expect.stringContaining("Informe: Benimaclet"),
        expect.stringContaining("Dispersión"),
        expect.stringContaining("Serie temporal"),
        expect.stringContaining("Mapa de anuncios"),
        expect.stringContaining("Gráfico de barras"),
        expect.stringContaining("Tabla"),
      ]),
    );
    // No interactive charts or maps inline in compact density (maps load lazily: give them time).
    await act(async () => new Promise((r) => setTimeout(r, 100)));
    expect(screen.queryByTestId("choropleth-mock")).toBeNull();
    expect(screen.queryByTestId("point-map-mock")).toBeNull();
    expect(screen.queryByTestId("plotly-mock")).toBeNull();
    expect(screen.queryByRole("img", { name: "Rentabilidad bruta (%)" })).toBeNull();
  });

  it("full_density_renders_every_block_inline", async () => {
    setState({ messages: [{ role: "user", content: "todo" }, assistant({ blocks: ALL_KINDS })] });
    render(<ChatPanel density="full" />);
    expect(screen.queryAllByRole("button", { name: /Ampliar/ })).toHaveLength(0);
    expect(await screen.findByTestId("choropleth-mock")).toBeInTheDocument();
    expect(await screen.findByTestId("point-map-mock")).toBeInTheDocument();
  });

  it("compact_card_opens_full_block_dialog", async () => {
    const map = ALL_KINDS.find((p) => p.kind === "choropleth")!;
    setState({ live: live({ text: "Mira el mapa", primitives: [map], status: "done" }) });
    render(<ChatPanel density="compact" />);
    const card = screen.getByRole("button", { name: /Mapa de barrios/ });
    expect(card).toHaveAttribute("aria-haspopup", "dialog");
    await act(async () => card.click());
    const dialog = screen.getByRole("dialog", { name: /Rentabilidad bruta \(%\)/ });
    expect(within(dialog).getByTestId("choropleth-mock")).toBeInTheDocument();
    await act(async () => within(dialog).getByRole("button", { name: "Cerrar" }).click());
    expect(screen.queryByRole("dialog")).toBeNull();
    // Focus goes back to the card that opened it.
    await waitFor(() => expect(card).toHaveFocus());
  });

  const pick = (kind: string, pred: (p: PrimitivePayload) => boolean = () => true) =>
    ALL_KINDS.find((p) => p.kind === kind && pred(p))!;
  it.each([
    ["bar", pick("bar")],
    ["line", pick("line")],
    ["scatter", pick("scatter")],
    ["plotly bar", pick("code", (p) => p.kind === "code" && p.output_type === "plotly_json")],
  ])("compact_card_sketches_%s", (_name, p) => {
    setState({ messages: [assistant({ blocks: [p] })] });
    render(<ChatPanel density="compact" />);
    const card = screen.getByRole("button", { name: /Ampliar/ });
    expect(card.querySelector("svg[preserveAspectRatio='none']")).not.toBeNull();
    expect(card.querySelector("svg rect, svg polyline, svg line")).not.toBeNull();
  });

  const plotly = (fig: unknown, raw?: string): PrimitivePayload => ({
    kind: "code",
    visualization_type: "plotly",
    code: "",
    fallback_kind: "table",
    output_type: "plotly_json",
    output: raw ?? JSON.stringify(fig),
  });
  it.each([
    ["lines", plotly({ data: [{ type: "scatter", mode: "lines", y: [1, 3, 2] }] }), "polyline", 1],
    ["markers", plotly({ data: [{ type: "scatter", mode: "markers", x: [1, 2, 3], y: [3, 1, 2] }] }), "line", 3],
    ["horizontal bars", plotly({ data: [{ type: "bar", orientation: "h", x: [1, 2], y: ["a", "b"] }] }), "rect", 2],
  ])("plotly_card_sketches_%s", (_name, p, shape, count) => {
    setState({ messages: [assistant({ blocks: [p] })] });
    render(<ChatPanel density="compact" />);
    const svg = screen.getByRole("button", { name: /Ampliar/ }).querySelector("svg[preserveAspectRatio='none']");
    expect(svg?.querySelectorAll(shape)).toHaveLength(count);
  });

  it.each([
    ["an unsupported trace", plotly({ data: [{ type: "heatmap", z: [[1]] }] })],
    ["output that is not JSON", plotly(null, "no es json")],
  ])("plotly_card_keeps_icon_for_%s", (_name, p) => {
    setState({ messages: [assistant({ blocks: [p] })] });
    render(<ChatPanel density="compact" />);
    expect(screen.getByRole("button", { name: /Ampliar/ }).querySelector("svg[preserveAspectRatio='none']")).toBeNull();
  });

  it.each([
    ["choropleth", /Mapa de barrios/, "choropleth-mock"],
    ["point_map", /Mapa de anuncios/, "point-map-mock"],
  ])("block_dialog_stays_on_same_block_when_stored_text_blocks_interleave_%s", async (_kind, card, mock) => {
    const pick = (k: string) => ALL_KINDS.find((p) => p.kind === k)!;
    const bar = pick("bar");
    const text = (content: string): PrimitivePayload => ({ kind: "text", content, role: "analysis" });
    // Live: streamed text is one block up front; stored: text blocks interleave with the figures.
    setState({ live: live({ text: "Mira", primitives: [bar, pick("choropleth"), pick("point_map")] }) }, true);
    const { rerender } = render(<ChatPanel density="compact" />);
    await act(async () => screen.getByRole("button", { name: card }).click());
    expect(within(screen.getByRole("dialog")).getByTestId(mock)).toBeInTheDocument();

    setState({
      messages: [
        { role: "user", content: "¿Dónde invierto?" },
        assistant({ blocks: [text("a"), bar, text("b"), pick("choropleth"), pick("point_map")] }),
      ],
    });
    await act(async () => rerender(<ChatPanel density="compact" />));
    expect(within(screen.getByRole("dialog")).getByTestId(mock)).toBeInTheDocument();
  });

  it("compact_header_actions", () => {
    const onExpand = vi.fn();
    const onClose = vi.fn();
    render(<ChatPanel density="compact" onExpand={onExpand} onClose={onClose} />);
    expect(screen.getByRole("heading", { name: "Rooster" })).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Abrir a pantalla completa" }));
    expect(onExpand).toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: "Cerrar chat" }));
    expect(onClose).toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: "Nueva conversación" }));
    expect(chat.reset).toHaveBeenCalled();
  });
});
