"use client";

import type { ReactNode } from "react";
import { AppShell } from "@/components/shell/AppShell";
import { useChatSurface } from "@/lib/chat-surface";
import type { UserOut } from "@/lib/types";
import { ChatSheet } from "./ChatSheet";
import { ChatSidebar } from "./ChatSidebar";

/** The signed-in shell with the chat beside the page: a sidebar from 1024 px, a sheet below. */
export function ChatSurfaces({
  user,
  conversations,
  children,
}: {
  user: UserOut;
  conversations: ReactNode;
  children: ReactNode;
}) {
  const s = useChatSurface();
  return (
    <>
      <AppShell
        user={user}
        conversations={conversations}
        chat={s.sidebarVisible ? <ChatSidebar /> : undefined}
        // On the full-screen route the chat is the page: no toggle.
        onToggleChat={s.onFullRoute ? undefined : s.toggle}
        chatOpen={s.isDesktop ? s.sidebarVisible : s.sheetOpen}
        floatingChat={!s.isDesktop && !s.onFullRoute}
      >
        {children}
      </AppShell>
      {!s.isDesktop && !s.onFullRoute ? <ChatSheet /> : null}
    </>
  );
}
