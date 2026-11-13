"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useRef, useState, type KeyboardEvent } from "react";
import { PencilIcon, Trash2Icon } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogClose,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Skeleton } from "@/components/ui/skeleton";
import { isStreaming, useChatContext } from "@/lib/chat-context";
import { conversationHref } from "@/lib/chat-navigation";
import { useChatSurface } from "@/lib/chat-surface";
import { COPY } from "@/lib/copy";
import { relativeDay } from "@/lib/format";
import type { ConversationSummary } from "@/lib/types";
import { useConversations } from "@/lib/use-conversations";
import { cn } from "@/lib/utils";

const NAV_MAX = 8;
/** Server limit for PATCH /conversations/{id}. */
const TITLE_MAX = 120;

function titleOf(c: ConversationSummary): string {
  return c.title?.trim() || COPY.newConversation;
}

/**
 * `nav`: the 8 most recent conversations (left nav). `recent`: the same list on Inicio. A click on
 * either opens it in the chat sidebar (sheet on phones). `page`: all of them, with date, rename and
 * delete (/chat); a click goes to the full-screen chat.
 * Switching is disabled while the chat is busy with a turn.
 */
export function ConversationList({ variant }: { variant: "nav" | "recent" | "page" }) {
  const { items, loading, rename, remove } = useConversations();
  const chat = useChatContext();
  const router = useRouter();
  const surface = useChatSurface();
  const activeId = chat.state.conversationId;
  const streaming = isStreaming(chat);
  const [editing, setEditing] = useState<{ id: string; value: string } | null>(null);
  /** The edit being committed or cancelled, so the blur that follows Enter/Esc is a no-op. */
  const editDone = useRef(false);
  const [deleting, setDeleting] = useState<ConversationSummary | null>(null);

  if (loading && items.length === 0) {
    return (
      <div aria-label={COPY.conversationsLoading} role="status" className="flex flex-col gap-1.5 px-2.5 py-1">
        <Skeleton className="h-4 w-3/4" />
        <Skeleton className="h-4 w-1/2" />
      </div>
    );
  }
  if (items.length === 0) {
    return (
      <p className="text-muted-foreground px-2.5 py-1 text-sm">
        {variant === "nav" ? COPY.navConversationsEmpty : COPY.homeRecentEmpty}
      </p>
    );
  }

  // Inicio and the nav show it beside the page; the list page goes to the full-screen route.
  const select = (id: string) => {
    void chat.open(id);
    if (variant === "page") router.push(conversationHref(id));
    else surface.show();
  };

  const itemButton = (c: ConversationSummary, className: string) => (
    <button
      type="button"
      onClick={() => select(c.id)}
      disabled={streaming}
      aria-current={c.id === activeId ? "true" : undefined}
      className={cn(
        "focus-visible:ring-ring/50 min-w-0 truncate rounded-md text-left outline-none focus-visible:ring-[3px] disabled:cursor-not-allowed disabled:opacity-60",
        className,
      )}
    >
      {titleOf(c)}
    </button>
  );

  if (variant !== "page") {
    return (
      <ul className="flex flex-col gap-0.5">
        {items.slice(0, NAV_MAX).map((c) => (
          <li key={c.id} className="flex min-w-0">
            {itemButton(
              c,
              "text-muted-foreground hover:bg-accent hover:text-foreground aria-[current=true]:bg-accent aria-[current=true]:text-foreground w-full px-2.5 py-1.5 text-sm",
            )}
          </li>
        ))}
        <li>
          <Link
            href="/chat"
            className="text-muted-foreground hover:text-foreground focus-visible:ring-ring/50 block rounded-md px-2.5 py-1.5 text-xs font-medium outline-none focus-visible:ring-[3px]"
          >
            {COPY.conversationsSeeAll}
          </Link>
        </li>
      </ul>
    );
  }

  const startEdit = (c: ConversationSummary) => {
    editDone.current = false;
    setEditing({ id: c.id, value: c.title ?? "" });
  };
  // Enter and blur save; Esc cancels. Whichever comes first ends the edit.
  const finishEdit = (save: boolean) => {
    if (!editing || editDone.current) return;
    editDone.current = true;
    setEditing(null);
    const title = editing.value.trim();
    const original = items.find((c) => c.id === editing.id);
    if (save && title && original && title !== original.title) void rename(editing.id, title);
  };
  const onEditKey = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter" || e.key === "Escape") {
      e.preventDefault();
      finishEdit(e.key === "Enter");
    }
  };

  const confirmDelete = () => {
    if (!deleting) return;
    const { id } = deleting;
    setDeleting(null);
    if (id === activeId) chat.reset();
    void remove(id);
  };

  return (
    <>
      <ul className="bg-card divide-y rounded-lg border">
        {items.map((c) => {
          const title = titleOf(c);
          // The conversation of a running turn cannot be renamed or deleted until the turn ends.
          const locked = streaming && c.id === activeId;
          return (
            <li key={c.id} className="flex min-w-0 items-center gap-2 px-3 py-2">
              {editing?.id === c.id ? (
                <Input
                  autoFocus
                  aria-label={COPY.conversationTitleLabel}
                  value={editing.value}
                  maxLength={TITLE_MAX}
                  onChange={(e) => setEditing({ id: c.id, value: e.target.value })}
                  onKeyDown={onEditKey}
                  onBlur={() => finishEdit(true)}
                  className="h-8 flex-1"
                />
              ) : (
                <>
                  {itemButton(
                    c,
                    "hover:text-foreground aria-[current=true]:text-foreground flex-1 px-1 py-1 text-sm font-medium aria-[current=true]:font-semibold",
                  )}
                  <span className="text-muted-foreground shrink-0 text-xs">{relativeDay(c.updated_at)}</span>
                </>
              )}
              <Button
                variant="ghost"
                size="icon-sm"
                aria-label={COPY.conversationRename.replace("{title}", title)}
                disabled={locked}
                onClick={() => startEdit(c)}
              >
                <PencilIcon aria-hidden />
              </Button>
              <Button
                variant="ghost"
                size="icon-sm"
                aria-label={COPY.conversationDelete.replace("{title}", title)}
                disabled={locked}
                onClick={() => setDeleting(c)}
              >
                <Trash2Icon aria-hidden />
              </Button>
            </li>
          );
        })}
      </ul>
      <Dialog open={deleting !== null} onOpenChange={(open) => !open && setDeleting(null)}>
        <DialogContent showCloseButton={false}>
          <DialogHeader>
            <DialogTitle>{COPY.conversationDeleteTitle}</DialogTitle>
            <DialogDescription>
              {COPY.conversationDeleteBody.replace("{title}", deleting ? titleOf(deleting) : "")}
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <DialogClose asChild>
              <Button variant="outline">{COPY.cancel}</Button>
            </DialogClose>
            <Button variant="destructive" onClick={confirmDelete}>
              {COPY.conversationDeleteConfirm}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
}
