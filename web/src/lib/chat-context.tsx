"use client";

import { createContext, useContext, type ReactNode } from "react";
import { useChat, type Chat } from "./use-chat";
import { useConversations } from "./use-conversations";

const ChatContext = createContext<Chat | null>(null);

/**
 * Holds the chat state in the (app) layout so a running turn survives client-side navigation.
 * Must sit inside <ConversationsProvider>: the list is refreshed when the chat changes it.
 */
export function ChatProvider({ userId, children }: { userId: string; children: ReactNode }) {
  const { refresh } = useConversations();
  const chat = useChat({ userId, onConversationsChanged: refresh });
  return <ChatContext.Provider value={chat}>{children}</ChatContext.Provider>;
}

export function useChatContext(): Chat {
  const chat = useContext(ChatContext);
  if (!chat) throw new Error("useChatContext must be used inside <ChatProvider>");
  return chat;
}

/** True while a turn is being streamed; switching or starting conversations is disabled then. */
export function isStreaming(chat: Chat): boolean {
  return chat.state.live?.status === "streaming";
}
