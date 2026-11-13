"use client";

// The signed-in user's conversation list. Rename and remove are optimistic and roll back
// (with a toast) when the API refuses; the chat hook calls `refresh` when the list changes.
import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { toast } from "sonner";
import { deleteConversation, listConversations, renameConversation, SignedOutError } from "./client";
import { COPY } from "./copy";
import type { ConversationSummary } from "./types";

export type Conversations = {
  items: ConversationSummary[];
  /** True until the first list has been fetched. */
  loading: boolean;
  refresh(): Promise<void>;
  rename(id: string, title: string): Promise<void>;
  remove(id: string): Promise<void>;
};

const ConversationsContext = createContext<Conversations | null>(null);

export function ConversationsProvider({ children }: { children: ReactNode }) {
  const [items, setItems] = useState<ConversationSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const current = useRef<ConversationSummary[]>([]);
  const update = useCallback((next: ConversationSummary[]) => {
    current.current = next;
    setItems(next);
  }, []);

  const refresh = useCallback(async () => {
    try {
      update(await listConversations());
    } catch {
      // Keep what is shown; the next refresh retries.
    } finally {
      setLoading(false);
    }
  }, [update]);

  // First load (written out rather than calling refresh(): setState only in the callbacks).
  useEffect(() => {
    let cancelled = false;
    listConversations()
      .then((list) => !cancelled && update(list))
      .catch(() => {})
      .finally(() => !cancelled && setLoading(false));
    return () => {
      cancelled = true;
    };
  }, [update]);

  const rename = useCallback(
    async (id: string, title: string) => {
      const previous = current.current.find((c) => c.id === id);
      if (!previous) return;
      update(current.current.map((c) => (c.id === id ? { ...c, title } : c)));
      try {
        const saved = await renameConversation(id, title);
        update(current.current.map((c) => (c.id === id ? saved : c)));
      } catch (err) {
        if (err instanceof SignedOutError) return;
        // Roll back only if no later rename has replaced this one.
        if (current.current.find((c) => c.id === id)?.title === title) {
          update(current.current.map((c) => (c.id === id ? previous : c)));
        }
        toast.error(COPY.conversationRenameError);
      }
    },
    [update],
  );

  const remove = useCallback(
    async (id: string) => {
      const index = current.current.findIndex((c) => c.id === id);
      if (index < 0) return;
      const removed = current.current[index];
      update(current.current.filter((c) => c.id !== id));
      try {
        await deleteConversation(id);
      } catch (err) {
        if (err instanceof SignedOutError) return;
        if (!current.current.some((c) => c.id === id)) {
          const next = [...current.current];
          next.splice(Math.min(index, next.length), 0, removed);
          update(next);
        }
        toast.error(COPY.conversationDeleteError);
      }
    },
    [update],
  );

  const value = useMemo(() => ({ items, loading, refresh, rename, remove }), [items, loading, refresh, rename, remove]);
  return <ConversationsContext.Provider value={value}>{children}</ConversationsContext.Provider>;
}

export function useConversations(): Conversations {
  const value = useContext(ConversationsContext);
  if (!value) throw new Error("useConversations must be used inside <ConversationsProvider>");
  return value;
}
