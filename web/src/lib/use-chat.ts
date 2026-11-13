"use client";

import { useCallback, useEffect, useReducer, useRef, useState } from "react";
import { chatReducer, initialChatState, type ChatState } from "./chat-state";
import { createConversation, getConversation, sendMessage, storageKey } from "./client";
import { COPY } from "./copy";
import { parseSSE } from "./sse";

/** After a 409, poll the conversation this often until the running turn is stored (~2 min max). */
const POLL_MS = 3000;
const POLL_MAX = 40;

export type UseChatOptions = {
  /** Owner of the conversations; the active id is remembered per user. */
  userId: string;
  /** Called after a conversation is created, after a turn finishes and when an opened one is gone. */
  onConversationsChanged?: () => void;
};

export type Chat = {
  state: ChatState;
  /** True from a send's start until it settles (stream and stored refetch); reset/open/startChat are ignored meanwhile. */
  busy: boolean;
  send: (text: string) => Promise<void>;
  /** Start a new, empty conversation. Ignored while a turn is streaming. */
  reset: () => void;
  /** Show another stored conversation. Ignored while a turn is streaming. */
  open: (id: string) => void;
  /** reset + send. Ignored while a turn is streaming. */
  startChat: (text: string) => Promise<void>;
};

export function useChat({ userId, onConversationsChanged }: UseChatOptions): Chat {
  const [state, dispatch] = useReducer(chatReducer, initialChatState);
  const key = storageKey(userId);
  const idRef = useRef<string | null>(null);
  const streamingRef = useRef(false);
  const [busy, setBusy] = useState(false);
  const abortRef = useRef<AbortController | null>(null);
  const changedRef = useRef(onConversationsChanged);
  useEffect(() => {
    changedRef.current = onConversationsChanged;
  });
  const notifyChanged = useCallback(() => changedRef.current?.(), []);
  /** Message count of the conversation as last read from the server (local-only failed turns excluded). */
  const storedCountRef = useRef(0);
  const stopPollRef = useRef<(() => void) | null>(null);

  const stopPoll = useCallback(() => {
    stopPollRef.current?.();
    stopPollRef.current = null;
  }, []);

  // Another turn is running on the server: refresh once it has been stored.
  const startPoll = useCallback(
    (id: string) => {
      stopPoll();
      const baseline = storedCountRef.current;
      let attempts = 0;
      let stopped = false;
      let timer: ReturnType<typeof setTimeout>;
      const tick = async () => {
        attempts++;
        const conv = await getConversation(id).catch(() => undefined);
        if (stopped) return;
        if (conv === null || (conv && conv.messages.length > baseline) || attempts >= POLL_MAX) {
          stopPollRef.current = null;
          if (conv && conv.messages.length > baseline) {
            storedCountRef.current = conv.messages.length;
            dispatch({ type: "finished", messages: conv.messages });
          }
          return;
        }
        timer = setTimeout(tick, POLL_MS);
      };
      timer = setTimeout(tick, POLL_MS);
      stopPollRef.current = () => {
        stopped = true;
        clearTimeout(timer);
      };
    },
    [stopPoll],
  );

  // Unmount (e.g. sign-out): stop polling and abandon a running send.
  useEffect(
    () => () => {
      stopPoll();
      abortRef.current?.abort();
    },
    [stopPoll],
  );

  // Restore the stored conversation; a stale id (404) is forgotten.
  useEffect(() => {
    let cancelled = false;
    const id = localStorage.getItem(key);
    if (!id) {
      dispatch({ type: "reset" });
      return;
    }
    getConversation(id)
      .then((conv) => {
        // A send that already created a conversation wins over the stored id.
        if (cancelled || idRef.current) return;
        if (conv) {
          idRef.current = conv.id;
          storedCountRef.current = conv.messages.length;
          dispatch({ type: "loaded", id: conv.id, messages: conv.messages });
        } else {
          localStorage.removeItem(key);
          dispatch({ type: "reset" });
        }
      })
      .catch(() => {
        // Network/server error: keep the id so the next send still targets it.
        if (cancelled || idRef.current) return;
        idRef.current = id;
        dispatch({ type: "loaded", id, messages: [] });
      });
    return () => {
      cancelled = true;
    };
  }, [key]);

  const send = useCallback(
    async (text: string) => {
      if (streamingRef.current) return;
      streamingRef.current = true;
      setBusy(true);
      stopPoll();
      const controller = new AbortController();
      abortRef.current = controller;
      const aborted = () => controller.signal.aborted;
      // Dispatched first so every failure below has a live turn with the user's text.
      dispatch({ type: "send", text });
      try {
        let id = idRef.current;
        if (!id) {
          id = await createConversation();
          if (aborted()) return;
          idRef.current = id;
          storedCountRef.current = 0;
          localStorage.setItem(key, id);
          dispatch({ type: "loaded", id, messages: [] });
          notifyChanged();
        }
        const res = await sendMessage(id, text, controller.signal);
        if (aborted()) return;
        if (res.status === 409) {
          dispatch({ type: "busy", text });
          startPoll(id);
          return;
        }
        if (res.status === 429) {
          dispatch({ type: "rateLimited", text });
          return;
        }
        if (!res.ok || !res.body) {
          dispatch({ type: "failed", reason: COPY.chatRequestFailed });
          return;
        }
        let sawDone = false;
        let sawError = false;
        for await (const event of parseSSE(res.body)) {
          if (aborted()) return;
          dispatch({ type: "event", event });
          if (event.type === "done") sawDone = true;
          if (event.type === "error") sawError = true;
        }
        if (aborted()) return;
        if (!sawDone) {
          dispatch({ type: "failed", reason: COPY.chatStreamInterrupted });
          return;
        }
        notifyChanged();
        // The `done` event already marked the live turn done. An errored turn stays
        // on screen as streamed; otherwise the stored conversation replaces it.
        if (sawError) return;
        const conv = await getConversation(id).catch(() => null);
        if (conv && !aborted()) {
          storedCountRef.current = conv.messages.length;
          dispatch({ type: "finished", messages: conv.messages });
        }
      } catch {
        if (!aborted()) dispatch({ type: "failed", reason: COPY.chatRequestFailed });
      } finally {
        if (abortRef.current === controller) abortRef.current = null;
        streamingRef.current = false;
        if (!aborted()) setBusy(false);
      }
    },
    [key, notifyChanged, startPoll, stopPoll],
  );

  const reset = useCallback(() => {
    if (streamingRef.current) return;
    stopPoll();
    idRef.current = null;
    localStorage.removeItem(key);
    dispatch({ type: "reset" });
  }, [key, stopPoll]);

  const open = useCallback(
    (id: string) => {
      if (streamingRef.current || idRef.current === id) return;
      stopPoll();
      idRef.current = id;
      storedCountRef.current = 0;
      localStorage.setItem(key, id);
      dispatch({ type: "open", id });
      getConversation(id)
        .then((conv) => {
          // Another conversation was opened (or started) meanwhile.
          if (idRef.current !== id) return;
          if (conv) {
            storedCountRef.current = conv.messages.length;
            dispatch({ type: "loaded", id, messages: conv.messages });
          } else {
            idRef.current = null;
            localStorage.removeItem(key);
            dispatch({ type: "reset" });
            notifyChanged();
          }
        })
        .catch(() => {
          // Network/server error: stay on it so the next send still targets it.
          if (idRef.current === id) dispatch({ type: "loaded", id, messages: [] });
        });
    },
    [key, notifyChanged, stopPoll],
  );

  const startChat = useCallback(
    async (text: string) => {
      if (streamingRef.current) return;
      reset();
      await send(text);
    },
    [reset, send],
  );

  return { state, busy, send, reset, open, startChat };
}
