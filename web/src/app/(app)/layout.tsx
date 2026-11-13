import { redirect } from "next/navigation";
import type { ReactNode } from "react";
import { ConversationList } from "@/components/chat/ConversationList";
import { ChatSurfaces } from "@/components/chat/ChatSurfaces";
import { Toaster } from "@/components/ui/sonner";
import { ChatProvider } from "@/lib/chat-context";
import { ChatSurfaceProvider } from "@/lib/chat-surface";
import { getCurrentUser } from "@/lib/server-api";
import { ConversationsProvider } from "@/lib/use-conversations";
import { PreferencesProvider } from "@/lib/use-preferences";

// Every signed-in page. The user (with stored preferences) is fetched here on the server;
// a session whose user the API no longer knows is signed out via /salir. The conversation
// list and the chat live here so a running turn survives client-side navigation, and every chat
// surface (sidebar, phone sheet, /chat/<id>) shows the same turn.
export default async function AppLayout({ children }: { children: ReactNode }) {
  const user = await getCurrentUser();
  if (!user) redirect("/salir");
  return (
    <PreferencesProvider initial={user.preferences}>
      <ConversationsProvider>
        <ChatProvider userId={user.id}>
          <ChatSurfaceProvider>
            <ChatSurfaces user={user} conversations={<ConversationList variant="nav" />}>
              {children}
            </ChatSurfaces>
          </ChatSurfaceProvider>
        </ChatProvider>
      </ConversationsProvider>
      <Toaster />
    </PreferencesProvider>
  );
}
