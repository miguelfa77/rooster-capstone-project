"use client";

import { MessageSquareIcon, SearchIcon } from "lucide-react";
import type { ReactNode } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { COPY } from "@/lib/copy";
import type { UserOut } from "@/lib/types";
import { MobileNav } from "./MobileNav";
import { UserMenu } from "./UserMenu";

type Props = {
  user: UserOut;
  /** Nav content for the mobile sheet. */
  nav: ReactNode;
  /** Chat toggle; the button stays disabled until a handler is given (Task 6). */
  onToggleChat?: () => void;
  chatOpen?: boolean;
};

export function TopBarV4({ user, nav, onToggleChat, chatOpen }: Props) {
  return (
    <header className="bg-card sticky top-0 z-40 flex h-14 items-center gap-2 border-b px-3 sm:gap-3 sm:px-6">
      <MobileNav>{nav}</MobileNav>
      {/* Search is enabled in Phase 2. */}
      <div role="search" className="relative max-w-[460px] min-w-0 flex-1">
        <SearchIcon aria-hidden className="text-muted-foreground pointer-events-none absolute top-1/2 left-3 size-4 -translate-y-1/2" />
        {/* The hint is an overlay so it can be shorter below sm (placeholder text cannot be responsive);
            `placeholder=" "` lets it hide once there is a value. */}
        <Input type="search" disabled aria-label={COPY.searchLabel} placeholder=" " className="peer pl-9" />
        <span
          aria-hidden
          className="text-placeholder pointer-events-none absolute inset-y-0 right-3 left-9 flex items-center truncate text-base peer-disabled:opacity-50 peer-[:not(:placeholder-shown)]:hidden md:text-sm"
        >
          <span className="sm:hidden">{`${COPY.searchLabel}…`}</span>
          <span className="hidden sm:inline">{COPY.searchPlaceholder}</span>
        </span>
      </div>
      <div className="ml-auto flex items-center gap-1 sm:gap-2">
        <Button
          variant="ghost"
          size="icon"
          aria-label={COPY.navChat}
          aria-pressed={onToggleChat ? Boolean(chatOpen) : undefined}
          disabled={!onToggleChat}
          onClick={onToggleChat}
        >
          <MessageSquareIcon aria-hidden />
        </Button>
        <UserMenu user={user} />
      </div>
    </header>
  );
}
