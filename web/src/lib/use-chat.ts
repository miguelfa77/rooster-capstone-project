"use client";

import { useCallback, useEffect, useReducer, useRef } from "react";
import { chatReducer, initialChatState, type ChatState } from "./chat-state";
import { createConversation, getConversation, sendMessage, STORAGE_KEY } from "./client";
import { COPY } from "./copy";
import { parseSSE } from "./sse";

export function useChat(): { state: ChatState; send: (text: string) => Promise<void>; reset: () => void } {
  const [state, dispatch] = useReducer(chatReducer, initialChatState);
  const idRef = useRef<string | null>(null);
  const streamingRef = useRef(false);
  const abortRef = useRef<AbortController | null>(null);

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
        localStorage.setItem(STORAGE_KEY, id);
        dispatch({ type: "loaded", id, messages: [] });
      }
      const res = await sendMessage(id, text, controller.signal);
      if (res.status === 409) {
        dispatch({ type: "busy" });
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
      if (conv && !controller.signal.aborted) dispatch({ type: "finished", messages: conv.messages });
    } catch {
      if (!controller.signal.aborted) dispatch({ type: "failed", reason: COPY.chatRequestFailed });
    } finally {
      if (abortRef.current === controller) {
        abortRef.current = null;
        streamingRef.current = false;
      }
    }
  }, []);

  const reset = useCallback(() => {
    abortRef.current?.abort();
    abortRef.current = null;
    streamingRef.current = false;
    idRef.current = null;
    localStorage.removeItem(STORAGE_KEY);
    dispatch({ type: "reset" });
  }, []);

  return { state, send, reset };
}
