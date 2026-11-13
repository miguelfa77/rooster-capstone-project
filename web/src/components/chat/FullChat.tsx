"use client";

import { useRouter } from "next/navigation";
import { useEffect, useRef } from "react";
import { useChatContext } from "@/lib/chat-context";
import { conversationHref } from "@/lib/chat-navigation";
import { useChatSurface } from "@/lib/chat-surface";
import { COPY } from "@/lib/copy";
import { useConversations } from "@/lib/use-conversations";
import { ChatPanel } from "./ChatPanel";

/** `/chat/<id>`: the chat as the page. A missing conversation goes to the list. */
export function FullChat({ id }: { id: string }) {
  const { open, state } = useChatContext();
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
    void open(id).then((found) => {
      if (!found && mounted.current) router.replace("/chat");
    });
  }, [id, open, router]);

  // A conversation started or switched to here (new conversation, nav) takes over the URL.
  const current = state.conversationId;
  const shown = useRef(false);
  useEffect(() => {
    if (current === id) shown.current = true;
    else if (current && shown.current) router.replace(conversationHref(current));
  }, [current, id, router]);

  const title = items.find((c) => c.id === current)?.title?.trim() || COPY.wordmark;
  return (
    // The viewport below the 56 px top bar; the message list scrolls inside.
    <div className="flex h-[calc(100svh-3.5rem)] min-h-0 flex-col">
      <ChatPanel density="full" title={title} onBack={back} />
    </div>
  );
}
