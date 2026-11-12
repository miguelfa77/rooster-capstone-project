// Pure chat state: the live (streaming) turn plus the stored history.
import { COPY } from "./copy";
import type { Message, PrimitivePayload, TurnEvent } from "./types";

export type LiveTurn = {
  userText: string;
  thoughts: string[];
  text: string;
  primitives: PrimitivePayload[];
  followUps: string[];
  error: { reason: string; suggestion: string } | null;
  status: "streaming" | "done";
};

export type ChatState = {
  conversationId: string | null;
  messages: Message[];
  live: LiveTurn | null;
  notice: string | null;
  loading: boolean;
};

export type ChatAction =
  | { type: "loaded"; id: string; messages: Message[] }
  | { type: "reset" }
  | { type: "send"; text: string }
  | { type: "event"; event: TurnEvent }
  | { type: "finished"; messages: Message[] }
  | { type: "failed"; reason: string }
  | { type: "busy" };

/** `loading` is true until the stored conversation (if any) has been checked. */
export const initialChatState: ChatState = {
  conversationId: null,
  messages: [],
  live: null,
  notice: null,
  loading: true,
};

function newLiveTurn(userText: string): LiveTurn {
  return { userText, thoughts: [], text: "", primitives: [], followUps: [], error: null, status: "streaming" };
}

function applyEvent(live: LiveTurn, event: TurnEvent): LiveTurn {
  switch (event.type) {
    case "thought":
      return { ...live, thoughts: [...live.thoughts, event.text] };
    case "text_delta":
      return { ...live, text: live.text + event.text };
    case "primitive":
      return { ...live, primitives: [...live.primitives, event.primitive] };
    case "follow_ups":
      return { ...live, followUps: event.items };
    case "error":
      return { ...live, error: { reason: event.reason, suggestion: event.suggestion } };
    case "done":
      return { ...live, status: "done" };
    default:
      return live;
  }
}

export function chatReducer(state: ChatState, action: ChatAction): ChatState {
  switch (action.type) {
    case "loaded":
      return { ...state, conversationId: action.id, messages: action.messages, loading: false };
    case "reset":
      return { ...initialChatState, loading: false };
    case "send":
      return { ...state, live: newLiveTurn(action.text), notice: null };
    case "event":
      return state.live ? { ...state, live: applyEvent(state.live, action.event) } : state;
    case "finished":
      return { ...state, messages: action.messages, live: null };
    case "failed": {
      const live = state.live ?? newLiveTurn("");
      return {
        ...state,
        live: { ...live, error: { reason: action.reason, suggestion: COPY.chatErrorSuggestion }, status: "done" },
      };
    }
    case "busy":
      return { ...state, live: null, notice: COPY.busy };
  }
}
