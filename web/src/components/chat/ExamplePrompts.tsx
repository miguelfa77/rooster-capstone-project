"use client";

import { isStreaming, useChatContext } from "@/lib/chat-context";
import { useChatSurface } from "@/lib/chat-surface";

/** Example questions; a click starts a new conversation with it and shows the chat beside the page. */
export function ExamplePrompts({ items }: { items: readonly string[] }) {
  const chat = useChatContext();
  const { show } = useChatSurface();
  const streaming = isStreaming(chat);
  return (
    <ul className="flex flex-wrap gap-2">
      {items.map((text) => (
        <li key={text} className="max-w-full">
          <button
            type="button"
            disabled={streaming}
            onClick={() => {
              void chat.startChat(text);
              show();
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
