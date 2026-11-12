"use client";

import { useCallback, useEffect, useReducer, useRef } from "react";
import { chatReducer, initialChatState, type ChatState } from "./chat-state";
import { createConversation, getConversation, sendMessage, STORAGE_KEY } from "./client";
import { COPY } from "./copy";
import { parseSSE } from "./sse";

/** After a 409, poll the conversation this often until the running turn is stored (~2 min max). */
const POLL_MS = 3000;
const POLL_MAX = 40;

export function useChat(): { state: ChatState; send: (text: string) => Promise<void>; reset: () => void } {
  const [state, dispatch] = useReducer(chatReducer, initialChatState);
  const idRef = useRef<string | null>(null);
  const streamingRef = useRef(false);
  const abortRef = useRef<AbortController | null>(null);
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

  useEffect(() => stopPoll, [stopPoll]);

  // Restore the stored conversation; a stale id (404) is forgotten.
  useEffect(() => {
    let cancelled = false;
    const id = localStorage.getItem(STORAGE_KEY);
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
          localStorage.removeItem(STORAGE_KEY);
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
  }, []);

  const send = useCallback(async (text: string) => {
    if (streamingRef.current) return;
    streamingRef.current = true;
    stopPoll();
    const controller = new AbortController();
    abortRef.current = controller;
    // Dispatched first so every failure below has a live turn with the user's text.
    dispatch({ type: "send", text });
    try {
      let id = idRef.current;
      if (!id) {
        id = await createConversation();
        if (controller.signal.aborted) return;
        idRef.current = id;
        storedCountRef.current = 0;
        localStorage.setItem(STORAGE_KEY, id);
        dispatch({ type: "loaded", id, messages: [] });
      }
      const res = await sendMessage(id, text, controller.signal);
      if (res.status === 409) {
        dispatch({ type: "busy", text });
        startPoll(id);
        return;
      }
      if (!res.ok || !res.body) {
        dispatch({ type: "failed", reason: COPY.chatRequestFailed });
        return;
      }
      let sawDone = false;
      let sawError = false;
      for await (const event of parseSSE(res.body)) {
        dispatch({ type: "event", event });
        if (event.type === "done") sawDone = true;
        if (event.type === "error") sawError = true;
      }
      if (!sawDone) {
        dispatch({ type: "failed", reason: COPY.chatStreamInterrupted });
        return;
      }
      // The `done` event already marked the live turn done. An errored turn stays
      // on screen as streamed; otherwise the stored conversation replaces it.
      if (sawError) return;
      const conv = await getConversation(id).catch(() => null);
      if (conv && !controller.signal.aborted) {
        storedCountRef.current = conv.messages.length;
        dispatch({ type: "finished", messages: conv.messages });
      }
    } catch {
      if (!controller.signal.aborted) dispatch({ type: "failed", reason: COPY.chatRequestFailed });
    } finally {
      if (abortRef.current === controller) {
        abortRef.current = null;
        streamingRef.current = false;
      }
    }
  }, [startPoll, stopPoll]);

  const reset = useCallback(() => {
    stopPoll();
    abortRef.current?.abort();
    abortRef.current = null;
    streamingRef.current = false;
    idRef.current = null;
    localStorage.removeItem(STORAGE_KEY);
    dispatch({ type: "reset" });
  }, [stopPoll]);

  return { state, send, reset };
}
