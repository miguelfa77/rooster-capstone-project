"use client";

import { ArrowLeftIcon, Maximize2Icon, SquarePenIcon, XIcon } from "lucide-react";
import { Button } from "@/components/ui/button";
import { COPY } from "@/lib/copy";
import type { Density } from "./AssistantMessage";

type Props = {
  density: Density;
  title: string;
  /** Starts a new conversation; disabled while a turn is running. */
  onNew: () => void;
  newDisabled: boolean;
  onExpand?: () => void;
  onClose?: () => void;
  onBack?: () => void;
};

/**
 * Compact (sidebar/sheet): title, then icon actions "Nueva conversación", expand and close.
 * Full (/chat/<id>): "Volver", title and "Nueva conversación".
 */
export function ChatHeader({ density, title, onNew, newDisabled, onExpand, onClose, onBack }: Props) {
  if (density === "full") {
    return (
      <header className="shrink-0 border-b">
        <div className="mx-auto flex h-12 w-full max-w-3xl items-center gap-2 px-2 sm:px-4">
          {onBack ? (
            <Button variant="ghost" size="sm" onClick={onBack}>
              <ArrowLeftIcon aria-hidden />
              {COPY.chatBack}
            </Button>
          ) : null}
          <h1 className="min-w-0 flex-1 truncate text-sm font-semibold">{title}</h1>
          <Button variant="outline" size="sm" onClick={onNew} disabled={newDisabled}>
            <SquarePenIcon aria-hidden />
            {COPY.newConversation}
          </Button>
        </div>
      </header>
    );
  }
  return (
    <header className="flex h-14 shrink-0 items-center gap-1 border-b pr-2 pl-4">
      <h2 className="min-w-0 flex-1 truncate text-[15px] font-semibold">{title}</h2>
      <Button variant="ghost" size="icon-sm" aria-label={COPY.newConversation} title={COPY.newConversation} onClick={onNew} disabled={newDisabled}>
        <SquarePenIcon aria-hidden />
      </Button>
      {onExpand ? (
        <Button variant="ghost" size="icon-sm" aria-label={COPY.chatExpand} title={COPY.chatExpand} onClick={onExpand}>
          <Maximize2Icon aria-hidden />
        </Button>
      ) : null}
      {onClose ? (
        <Button variant="ghost" size="icon-sm" aria-label={COPY.chatCloseSidebar} title={COPY.chatCloseSidebar} onClick={onClose}>
          <XIcon aria-hidden />
        </Button>
      ) : null}
    </header>
  );
}
