"use client";

// Which chat surface is showing: the right sidebar (from 1024 px, while `prefs.chat_open` is not
// false), the bottom sheet (below 1024 px) or the full-screen route `/chat/<id>` (`/chat/nueva`
// while there is no conversation yet). All three render the one chat state held by <ChatProvider>.
// Where the user reads a conversation is decided here and only here: from Inicio and the nav the
// chat opens beside the page instead of navigating; the "expand" actions go to the full-screen route.
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { useChatContext } from "./chat-context";
import { useIsDesktop } from "./use-media-query";
import { usePreferences } from "./use-preferences";

/** The full-screen route of a conversation, or the empty full-screen chat when there is none. */
export function conversationHref(id: string | null): string {
  return id ? `/chat/${encodeURIComponent(id)}` : "/chat/nueva";
}

/** `/chat/<id>` and `/chat/nueva`: the chat is the page, so the sidebar and sheet are not shown. */
function isFullChatRoute(pathname: string): boolean {
  return /^\/chat\/[^/]+/.test(pathname);
}

export type ChatSurface = {
  isDesktop: boolean;
  /** On `/chat/<id>` or `/chat/nueva`: the chat is the page. */
  onFullRoute: boolean;
  sidebarVisible: boolean;
  sheetOpen: boolean;
  setSheetOpen(open: boolean): void;
  /** The top-bar chat button: sidebar on/off, or the sheet below 1024 px. */
  toggle(): void;
  closeSidebar(): void;
  /** Show the chat where the user is (after startChat/open from Inicio or the nav). */
  show(): void;
  /** Full-screen route of the current conversation (the empty full-screen chat when there is none). */
  expand(): void;
  /** From the full-screen route back to the previous page, with the chat open. */
  back(): void;
};

const ChatSurfaceContext = createContext<ChatSurface | null>(null);

/** Must sit inside <PreferencesProvider> and <ChatProvider>. */
export function ChatSurfaceProvider({ children }: { children: ReactNode }) {
  const isDesktop = useIsDesktop();
  const pathname = usePathname();
  const router = useRouter();
  const search = useSearchParams().toString();
  const { prefs, set } = usePreferences();
  const { state } = useChatContext();
  const [sheetWanted, setSheetOpen] = useState(false);
  const onFullRoute = isFullChatRoute(pathname);
  const chatOpen = prefs.chat_open !== false;
  const sidebarVisible = isDesktop && !onFullRoute && chatOpen;
  // A sheet left open while widening the window is not shown on desktop.
  const sheetOpen = sheetWanted && !isDesktop && !onFullRoute;

  // The last page that was not the full-screen chat, for "Volver".
  const previous = useRef<string | null>(null);
  useEffect(() => {
    if (!onFullRoute) previous.current = search ? `${pathname}?${search}` : pathname;
  }, [pathname, search, onFullRoute]);

  const openSidebar = useCallback(() => {
    if (!chatOpen) set({ chat_open: true });
  }, [chatOpen, set]);
  const show = useCallback(() => {
    if (isDesktop) openSidebar();
    else setSheetOpen(true);
  }, [isDesktop, openSidebar]);
  const toggle = useCallback(() => {
    if (isDesktop) set({ chat_open: !chatOpen });
    else setSheetOpen((o) => !o);
  }, [isDesktop, chatOpen, set]);
  const closeSidebar = useCallback(() => set({ chat_open: false }), [set]);
  const conversationId = state.conversationId;
  const expand = useCallback(() => {
    setSheetOpen(false);
    router.push(conversationHref(conversationId));
  }, [router, conversationId]);
  const back = useCallback(() => {
    show();
    router.push(previous.current ?? "/");
  }, [router, show]);

  const value = useMemo(
    () => ({ isDesktop, onFullRoute, sidebarVisible, sheetOpen, setSheetOpen, toggle, closeSidebar, show, expand, back }),
    [isDesktop, onFullRoute, sidebarVisible, sheetOpen, toggle, closeSidebar, show, expand, back],
  );
  return <ChatSurfaceContext.Provider value={value}>{children}</ChatSurfaceContext.Provider>;
}

export function useChatSurface(): ChatSurface {
  const value = useContext(ChatSurfaceContext);
  if (!value) throw new Error("useChatSurface must be used inside <ChatSurfaceProvider>");
  return value;
}
