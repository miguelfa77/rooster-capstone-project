"use client";

import type { ReactNode } from "react";
import type { UserOut } from "@/lib/types";
import { cn } from "@/lib/utils";
import { LeftNav } from "./LeftNav";
import { TopBarV4 } from "./TopBarV4";

type Props = {
  user: UserOut;
  children: ReactNode;
  /** Right-hand chat column (Task 6). */
  chat?: ReactNode;
  /** Content of the nav's "Conversaciones" section (Task 5). */
  conversations?: ReactNode;
  onToggleChat?: () => void;
  chatOpen?: boolean;
};

/** Signed-in layout: 220 px nav (a sheet below 1024 px), main column, optional chat column. */
export function AppShell({ user, children, chat, conversations, onToggleChat, chatOpen }: Props) {
  const nav = <LeftNav conversations={conversations} />;
  return (
    <div
      className={cn(
        "min-h-svh lg:grid",
        chat ? "lg:grid-cols-[220px_minmax(0,1fr)_340px]" : "lg:grid-cols-[220px_minmax(0,1fr)]",
      )}
    >
      <aside className="bg-card sticky top-0 hidden h-svh overflow-y-auto border-r lg:block">{nav}</aside>
      <div className="flex min-w-0 flex-col">
        <TopBarV4 user={user} nav={nav} onToggleChat={onToggleChat} chatOpen={chatOpen} />
        <main className="min-w-0 flex-1">{children}</main>
      </div>
      {chat && <div className="min-w-0">{chat}</div>}
    </div>
  );
}
