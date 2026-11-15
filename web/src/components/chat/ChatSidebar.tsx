"use client";

import { useChatSurface } from "@/lib/chat-surface";
import { COPY } from "@/lib/copy";
import { ChatPanel } from "./ChatPanel";

/**
 * The right-hand chat column (380 px, from 1024 px). It stays in view while the page scrolls.
 * `hidden lg:flex` keeps it off phones even before hydration has measured the viewport.
 */
export function ChatSidebar() {
  const { expand, closeSidebar } = useChatSurface();
  return (
    <aside
      aria-label={COPY.chatSidebarLabel}
      className="bg-card sticky top-0 hidden h-svh min-w-0 flex-col border-l lg:flex"
    >
      <ChatPanel density="compact" onExpand={expand} onClose={closeSidebar} />
    </aside>
  );
}
