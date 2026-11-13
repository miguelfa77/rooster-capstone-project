// Where the user reads a conversation. One place for the decision: from Inicio and the nav the
// chat opens beside the page (the sidebar from 1024 px, the bottom sheet below) instead of
// navigating; the /chat list page and the "expand" actions go to the full-screen route.

export type ChatSurfaceTarget = {
  isDesktop: boolean;
  openSidebar(): void;
  openSheet(): void;
};

/** After startChat/open from Inicio or the nav: show the chat where the user already is. */
export function showConversation(target: ChatSurfaceTarget): void {
  if (target.isDesktop) target.openSidebar();
  else target.openSheet();
}

/** The full-screen route of a conversation, or the list when there is none. */
export function conversationHref(id: string | null): string {
  return id ? `/chat/${encodeURIComponent(id)}` : "/chat";
}

/** `/chat/<id>`: the chat is the page, so the sidebar and sheet are not shown. */
export function isFullChatRoute(pathname: string): boolean {
  return /^\/chat\/[^/]+/.test(pathname);
}
