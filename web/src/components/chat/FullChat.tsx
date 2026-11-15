"use client";

import { useRouter } from "next/navigation";
import { useEffect, useRef } from "react";
import { useChatContext } from "@/lib/chat-context";
import { conversationHref, useChatSurface } from "@/lib/chat-surface";
import { COPY } from "@/lib/copy";
import { useConversations } from "@/lib/use-conversations";
import { ChatPanel } from "./ChatPanel";

/**
 * `/chat/<id>` (or `/chat/nueva` for `id` null): the chat as the page. A missing conversation goes
 * to the list. The URL always follows the conversation the chat is showing.
 */
export function FullChat({ id }: { id: string | null }) {
  const { open, reset, state, busy } = useChatContext();
  const { items } = useConversations();
  const { back } = useChatSurface();
  const router = useRouter();

  // Not a cancel-on-cleanup flag: a dev-mode effect re-run must not drop the 404 redirect.
  const mounted = useRef(false);
  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
    };
  }, []);

  useEffect(() => {
    if (id === null) {
      reset();
      return;
    }
    void open(id).then((found) => {
      if (!found && mounted.current) router.replace("/chat");
    });
  }, [id, open, reset, router]);

  const current = state.conversationId;
  const shown = useRef(false);
  useEffect(() => {
    if (!current) return;
    if (id === null) {
      // The first message created the conversation: take over the URL in place. router.replace
      // would swap this page for /chat/[id] and remount the composer mid-send.
      if (window.location.pathname !== conversationHref(current)) {
        window.history.replaceState(null, "", conversationHref(current));
      }
      return;
    }
    if (current === id) shown.current = true;
    // A conversation started or switched to meanwhile (new conversation, nav) takes over the URL; so
    // does the running one when history navigation lands on another id (opening is refused while busy).
    else if (shown.current || busy) router.replace(conversationHref(current));
  }, [current, id, busy, router]);

  const title = items.find((c) => c.id === current)?.title?.trim() || COPY.wordmark;
  return (
    // The viewport below the 56 px top bar; the message list scrolls inside.
    <div className="flex h-[calc(100svh-3.5rem)] min-h-0 flex-col">
      <ChatPanel
        density="full"
        title={title}
        onBack={back}
        onNewConversation={() => router.replace(conversationHref(null))}
      />
    </div>
  );
}
