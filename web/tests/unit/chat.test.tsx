import { fireEvent, render, screen, within } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { ChatState, LiveTurn } from "@/lib/chat-state";
import type { AssistantMessage, Message } from "@/lib/types";

vi.mock("@/components/map/ChoroplethMap", () => ({ default: () => <div data-testid="choropleth-mock" /> }));
vi.mock("@/components/map/PointMap", () => ({ default: () => <div data-testid="point-map-mock" /> }));

const chat = vi.hoisted(() => ({
  state: {} as ChatState,
  send: vi.fn<(text: string) => Promise<void>>(async () => {}),
  reset: vi.fn(),
}));
vi.mock("@/lib/use-chat", () => ({ useChat: () => chat }));

import ChatView from "@/components/chat/ChatView";

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

function setState(over: Partial<ChatState>) {
  chat.state = { conversationId: "c1", messages: [], live: null, notice: null, loading: false, ...over };
}

const input = () => screen.getByPlaceholderText("Pregunta por anuncios, rentabilidades, barrios…");

beforeEach(() => {
  chat.send.mockClear();
  chat.reset.mockClear();
  setState({});
});

describe("ChatView", () => {
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

  it("composer_disabled_while_streaming", () => {
    setState({ live: live() });
    const { rerender } = render(<ChatView />);
    expect(input()).toBeDisabled();
    expect(screen.getByRole("button", { name: "Enviar" })).toBeDisabled();

    setState({ loading: true, conversationId: null });
    rerender(<ChatView />);
    expect(input()).toBeDisabled();

    setState({ live: live({ status: "done", error: { reason: "r", suggestion: "s" } }) });
    rerender(<ChatView />);
    expect(input()).toBeEnabled();
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
