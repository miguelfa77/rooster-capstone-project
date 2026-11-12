"use client";

import { createContext, useContext, type ReactNode } from "react";
import { useChat } from "./use-chat";

type Chat = ReturnType<typeof useChat>;

const ChatContext = createContext<Chat | null>(null);

/** Holds the chat state in the root layout so a running turn survives client-side navigation. */
export function ChatProvider({ children }: { children: ReactNode }) {
  const chat = useChat();
  return <ChatContext.Provider value={chat}>{children}</ChatContext.Provider>;
}

export function useChatContext(): Chat {
  const chat = useContext(ChatContext);
  if (!chat) throw new Error("useChatContext must be used inside <ChatProvider>");
  return chat;
}
