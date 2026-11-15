"use client";

import { useChatContext } from "@/lib/chat-context";
import { COPY } from "@/lib/copy";
import { cn } from "@/lib/utils";
import type { Density } from "./AssistantMessage";
import { ChatHeader } from "./ChatHeader";
import Composer from "./Composer";
import MessageList from "./MessageList";

function EmptyChat({ density, onPick, disabled }: { density: Density; onPick: (text: string) => void; disabled: boolean }) {
  const full = density === "full";
  return (
    <div
      data-testid="chat-empty"
      className={cn(
        "flex min-h-0 flex-1 flex-col justify-center gap-3 overflow-y-auto",
        full ? "mx-auto w-full max-w-3xl items-center px-4 text-center" : "px-4 py-6",
      )}
    >
      {full ? <p className="text-3xl font-semibold tracking-tight">{COPY.wordmark}</p> : null}
      <p className="text-muted-foreground text-sm">{COPY.tagline}</p>
      <div className={cn("flex flex-wrap gap-2", full && "justify-center")}>
        {COPY.chatExamples.map((q) => (
          <button
            key={q}
            type="button"
            disabled={disabled}
            onClick={() => onPick(q)}
            className="bg-card hover:bg-accent focus-visible:ring-ring/50 max-w-full rounded-2xl border px-3 py-1.5 text-left text-[13px] outline-none [overflow-wrap:anywhere] focus-visible:ring-[3px] disabled:cursor-not-allowed disabled:opacity-60"
          >
            {q}
          </button>
        ))}
      </div>
    </div>
  );
}

type Props = {
  /** `compact`: sidebar and phone sheet (big blocks as cards). `full`: the full-screen route. */
  density: Density;
  /** Header title; defaults to the wordmark. */
  title?: string;
  onExpand?: () => void;
  onClose?: () => void;
  onBack?: () => void;
  /** Runs after a new conversation was started (the full-screen route leaves the old id's URL). */
  onNewConversation?: () => void;
};

/** The one chat UI: header, history with the live turn, notice and composer. Fills its parent's height. */
export function ChatPanel({ density, title = COPY.wordmark, onExpand, onClose, onBack, onNewConversation }: Props) {
  const { state, busy, send, reset } = useChatContext();
  const { messages, live, notice, draft, loading } = state;
  const disabled = busy || loading;
  const isEmpty = !loading && messages.length === 0 && !live;
  const onSend = (text: string) => void send(text);

  return (
    <div className="flex h-full min-h-0 min-w-0 flex-col">
      <ChatHeader
        density={density}
        title={title}
        onNew={() => {
          reset();
          onNewConversation?.();
        }}
        newDisabled={busy}
        onExpand={onExpand}
        onClose={onClose}
        onBack={onBack}
      />
      {isEmpty ? (
        <EmptyChat density={density} onPick={onSend} disabled={disabled} />
      ) : (
        <MessageList messages={messages} live={live} onFollowUp={onSend} density={density} />
      )}
      <div className={cn("shrink-0", density === "full" ? "mx-auto w-full max-w-3xl px-4 pb-4 sm:px-6" : "px-3.5 pb-3.5")}>
        {notice ? (
          <p role="status" className="bg-card text-muted-foreground mb-2 rounded-lg border px-3 py-2 text-sm">
            {notice}
          </p>
        ) : null}
        <Composer disabled={disabled} draft={draft} onSend={onSend} />
      </div>
    </div>
  );
}
