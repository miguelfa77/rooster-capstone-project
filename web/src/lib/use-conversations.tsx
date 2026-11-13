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

/** Most recently updated first, as the API lists them. */
function byUpdatedDesc(items: ConversationSummary[]): ConversationSummary[] {
  return [...items].sort((a, b) => Date.parse(b.updated_at) - Date.parse(a.updated_at));
}

export function ConversationsProvider({ children }: { children: ReactNode }) {
  const [items, setItems] = useState<ConversationSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const current = useRef<ConversationSummary[]>([]);
  const update = useCallback((next: ConversationSummary[]) => {
    current.current = next;
    setItems(next);
  }, []);

  // Every list request takes a number; only the latest one started may apply its answer.
  const seq = useRef(0);
  const load = useCallback(() => {
    const mine = ++seq.current;
    return listConversations()
      .then((list) => {
        if (mine === seq.current) update(list);
      })
      .catch(() => {
        // Keep what is shown; the next refresh retries.
      })
      .finally(() => {
        if (mine === seq.current) setLoading(false);
      });
  }, [update]);

  const refresh = useCallback(() => load(), [load]);

  useEffect(() => {
    void load();
  }, [load]);

  const rename = useCallback(
    async (id: string, title: string) => {
      const previous = current.current.find((c) => c.id === id);
      if (!previous) return;
      update(current.current.map((c) => (c.id === id ? { ...c, title } : c)));
      try {
        const saved = await renameConversation(id, title);
        update(byUpdatedDesc(current.current.map((c) => (c.id === id ? saved : c))));
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
