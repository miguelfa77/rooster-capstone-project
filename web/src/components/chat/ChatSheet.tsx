"use client";

import { MessageSquareIcon } from "lucide-react";
import { useRef } from "react";
import { Button } from "@/components/ui/button";
import { Sheet, SheetContent, SheetTitle, SheetTrigger } from "@/components/ui/sheet";
import { useChatSurface } from "@/lib/chat-surface";
import { COPY } from "@/lib/copy";
import { ChatPanel } from "./ChatPanel";

/** Below 1024 px: a floating button opens the chat as a bottom sheet (85 % of the height). */
export function ChatSheet() {
  const { sheetOpen, setSheetOpen, expand } = useChatSurface();
  // The sheet opens from the floating button, the top bar or Inicio; focus returns to whichever it was.
  const opener = useRef<HTMLElement | null>(null);
  return (
    <Sheet open={sheetOpen} onOpenChange={setSheetOpen}>
      <SheetTrigger asChild>
        <Button
          size="icon-lg"
          aria-label={COPY.chatOpenSheet}
          className="fixed right-4 bottom-4 z-40 rounded-full shadow-lg lg:hidden"
        >
          <MessageSquareIcon aria-hidden />
        </Button>
      </SheetTrigger>
      <SheetContent
        side="bottom"
        showCloseButton={false}
        aria-describedby={undefined}
        className="bg-card h-[85svh] gap-0 overflow-hidden rounded-t-xl p-0 focus:outline-none"
        // Focus the sheet itself, so the phone keyboard does not pop up on open.
        onOpenAutoFocus={(e) => {
          opener.current = document.activeElement instanceof HTMLElement ? document.activeElement : null;
          e.preventDefault();
          (e.currentTarget as HTMLElement).focus();
        }}
        // (Without a focused opener, e.g. Safari does not focus clicked buttons, Radix focuses the floating button.)
        onCloseAutoFocus={(e) => {
          const el = opener.current;
          opener.current = null;
          if (el && el !== document.body && el.isConnected) {
            e.preventDefault();
            el.focus();
          }
        }}
      >
        <SheetTitle className="sr-only">{COPY.chatSidebarLabel}</SheetTitle>
        <ChatPanel density="compact" onExpand={expand} onClose={() => setSheetOpen(false)} />
      </SheetContent>
    </Sheet>
  );
}
