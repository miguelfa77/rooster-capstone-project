"use client";

import type { ReactNode } from "react";
import type { UserOut } from "@/lib/types";
import { cn } from "@/lib/utils";
import { LeftNav } from "./LeftNav";
import { TopBarV4 } from "./TopBarV4";

type Props = {
  user: UserOut;
  children: ReactNode;
  /** Right-hand chat column (380 px, from 1024 px); rendered as the grid's third column. */
  chat?: ReactNode;
  /** Content of the nav's "Conversaciones" section (Task 5). */
  conversations?: ReactNode;
  onToggleChat?: () => void;
  chatOpen?: boolean;
  /** The floating chat button is shown below 1024 px: keep the end of the page clear of it. */
  floatingChat?: boolean;
};

/**
 * Signed-in layout: 220 px nav (a sheet below 1024 px), main column, optional 380 px chat column.
 * The main column is `minmax(0, 1fr)` so the chat never pushes the page into horizontal scroll.
 */
export function AppShell({ user, children, chat, conversations, onToggleChat, chatOpen, floatingChat }: Props) {
  const nav = <LeftNav conversations={conversations} />;
  return (
    <div
      className={cn(
        "min-h-svh lg:grid",
        chat ? "lg:grid-cols-[220px_minmax(0,1fr)_380px]" : "lg:grid-cols-[220px_minmax(0,1fr)]",
      )}
    >
      <aside className="bg-card sticky top-0 hidden h-svh overflow-y-auto border-r lg:block">{nav}</aside>
      <div className="flex min-w-0 flex-col">
        <TopBarV4 user={user} nav={nav} onToggleChat={onToggleChat} chatOpen={chatOpen} />
        <main className={cn("min-w-0 flex-1", floatingChat && "pb-20 lg:pb-0")}>{children}</main>
      </div>
      {chat}
    </div>
  );
}
