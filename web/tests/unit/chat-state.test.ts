import { describe, expect, it } from "vitest";
import { chatReducer, initialChatState, type ChatState, type LiveTurn } from "@/lib/chat-state";
import { COPY } from "@/lib/copy";
import type { Message, PrimitivePayload } from "@/lib/types";

const idle: ChatState = { ...initialChatState, conversationId: "c1", loading: false };

function liveTurn(overrides: Partial<LiveTurn> = {}): LiveTurn {
  return {
    userText: "hola",
    thoughts: [],
    text: "",
    primitives: [],
    followUps: [],
    error: null,
    status: "streaming",
    ...overrides,
  };
}

const streaming: ChatState = { ...idle, live: liveTurn() };

describe("chatReducer", () => {
  it("send_creates_live_turn", () => {
    const s = chatReducer({ ...idle, notice: "old" }, { type: "send", text: "¿Dónde invertir?" });
    expect(s.live).toEqual(liveTurn({ userText: "¿Dónde invertir?" }));
    expect(s.notice).toBeNull();
  });

  it("send_after_errored_turn_keeps_it_in_messages", () => {
    const prior: Message[] = [{ role: "user", content: "antes" }];
    const errored = liveTurn({ text: "parcial", error: { reason: "Se cortó.", suggestion: "Reintenta." }, status: "done" });
    const s = chatReducer({ ...idle, messages: prior, live: errored }, { type: "send", text: "otra" });
    expect(s.messages).toEqual([
      ...prior,
      { role: "user", content: "hola" },
      {
        role: "assistant",
        blocks: [],
        summary: "",
        primitive_payloads: [],
        follow_ups: [],
        graceful_fallback: { reason: "Se cortó.", suggestion: "Reintenta." },
      },
    ]);
    expect(s.live).toEqual(liveTurn({ userText: "otra" }));
  });

  it("send_after_successful_live_turn_does_not_append", () => {
    const s = chatReducer({ ...idle, live: liveTurn({ status: "done" }) }, { type: "send", text: "otra" });
    expect(s.messages).toEqual([]);
  });

  it("thought_appends", () => {
    let s = chatReducer(streaming, { type: "event", event: { type: "thought", text: "a" } });
    s = chatReducer(s, { type: "event", event: { type: "thought", text: "b" } });
    expect(s.live?.thoughts).toEqual(["a", "b"]);
  });

  it("text_delta_concatenates", () => {
    let s = chatReducer(streaming, { type: "event", event: { type: "text_delta", text: "Ruz" } });
    s = chatReducer(s, { type: "event", event: { type: "text_delta", text: "afa" } });
    expect(s.live?.text).toBe("Ruzafa");
  });

  it("primitive_appends", () => {
    const p1: PrimitivePayload = { kind: "text", content: "uno", role: "lead" };
    const p2: PrimitivePayload = { kind: "text", content: "dos", role: "analysis" };
    let s = chatReducer(streaming, { type: "event", event: { type: "primitive", primitive: p1 } });
    s = chatReducer(s, { type: "event", event: { type: "primitive", primitive: p2 } });
    expect(s.live?.primitives).toEqual([p1, p2]);
  });

  it("follow_ups_set", () => {
    const s = chatReducer(streaming, { type: "event", event: { type: "follow_ups", items: ["x", "y"] } });
    expect(s.live?.followUps).toEqual(["x", "y"]);
  });

  it("error_event_sets_error", () => {
    const s = chatReducer(streaming, {
      type: "event",
      event: { type: "error", reason: "boom", suggestion: "otra vez" },
    });
    expect(s.live?.error).toEqual({ reason: "boom", suggestion: "otra vez" });
  });

  it("done_marks_done", () => {
    const s = chatReducer(streaming, { type: "event", event: { type: "done" } });
    expect(s.live?.status).toBe("done");
  });

  it("finished_replaces_messages_and_clears_live", () => {
    const messages: Message[] = [
      { role: "user", content: "hola" },
      { role: "assistant", blocks: [], summary: "", primitive_payloads: [], follow_ups: [] },
    ];
    const s = chatReducer(
      { ...streaming, messages: [{ role: "user", content: "viejo" }] },
      { type: "finished", messages },
    );
    expect(s.messages).toEqual(messages);
    expect(s.live).toBeNull();
  });

  it("failed_sets_fallback", () => {
    const s = chatReducer(streaming, { type: "failed", reason: "sin conexión" });
    expect(s.live?.error).toEqual({ reason: "sin conexión", suggestion: COPY.chatErrorSuggestion });
    expect(s.live?.status).toBe("done");
    expect(s.live?.userText).toBe("hola");

    // No live turn (e.g. conversation creation failed): the error must still be visible.
    const s2 = chatReducer(idle, { type: "failed", reason: "sin conexión" });
    expect(s2.live?.error).toEqual({ reason: "sin conexión", suggestion: COPY.chatErrorSuggestion });
    expect(s2.live?.status).toBe("done");
  });

  it("busy_sets_notice_without_message", () => {
    const s = chatReducer(streaming, { type: "busy", text: "hola" });
    expect(s.notice).toBe(COPY.busy);
    expect(s.draft).toEqual({ text: "hola" });
    expect(s.live).toBeNull();
    expect(s.messages).toEqual(streaming.messages);
  });

  it("finished_clears_busy_notice_and_draft", () => {
    const busy = chatReducer(streaming, { type: "busy", text: "hola" });
    const s = chatReducer(busy, { type: "finished", messages: [] });
    expect(s.notice).toBeNull();
    expect(s.draft).toBeNull();
  });
});
