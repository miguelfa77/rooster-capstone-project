"use client";

import { useRouter } from "next/navigation";
import { useEffect, useRef } from "react";
import { isStreaming, useChatContext } from "@/lib/chat-context";
import { showConversation } from "@/lib/chat-navigation";

/** Example questions; a click starts a new conversation with it and shows it once it exists. */
export function ExamplePrompts({ items }: { items: readonly string[] }) {
  const chat = useChatContext();
  const router = useRouter();
  const pending = useRef(false);
  const { conversationId, live } = chat.state;

  useEffect(() => {
    if (!pending.current) return;
    if (conversationId) {
      pending.current = false;
      showConversation(router, conversationId);
    } else if (live?.error) {
      // Creating the conversation failed; nothing to show.
      pending.current = false;
    }
  }, [conversationId, live?.error, router]);

  const streaming = isStreaming(chat);
  return (
    <ul className="flex flex-wrap gap-2">
      {items.map((text) => (
        <li key={text} className="max-w-full">
          <button
            type="button"
            disabled={streaming}
            onClick={() => {
              pending.current = true;
              // Settles at the end of the turn, or at once if the start was ignored.
              void chat.startChat(text).finally(() => (pending.current = false));
            }}
            className="bg-card hover:bg-accent focus-visible:ring-ring/50 block max-w-full rounded-2xl border sm:rounded-full px-3.5 py-2 text-left text-sm outline-none focus-visible:ring-[3px] disabled:cursor-not-allowed disabled:opacity-60"
          >
            {text}
          </button>
        </li>
      ))}
    </ul>
  );
}
