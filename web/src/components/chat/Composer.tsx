"use client";

import { ArrowUpIcon } from "lucide-react";
import { useState, type KeyboardEvent } from "react";
import { Button } from "@/components/ui/button";
import { COPY } from "@/lib/copy";

const MAX_CHARS = 2000;
const COUNTER_FROM = 1800;

/** Message input: Enter sends, Shift+Enter inserts a newline. A new `draft` (a refused send) refills it. */
export default function Composer({
  disabled,
  draft = null,
  onSend,
}: {
  disabled: boolean;
  draft?: { text: string } | null;
  onSend: (text: string) => void;
}) {
  const [text, setText] = useState(draft?.text ?? "");
  const [seenDraft, setSeenDraft] = useState(draft);
  if (draft !== seenDraft) {
    setSeenDraft(draft);
    if (draft) setText(draft.text);
  }
  const canSend = !disabled && text.trim() !== "";

  function submit() {
    if (!canSend) return;
    onSend(text.trim());
    setText("");
  }

  function onKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key !== "Enter" || e.shiftKey || e.nativeEvent.isComposing) return;
    e.preventDefault();
    submit();
  }

  return (
    <form
      onSubmit={(e) => {
        e.preventDefault();
        submit();
      }}
      className="bg-card border-input focus-within:border-ring focus-within:ring-ring/50 has-[textarea:disabled]:opacity-60 flex items-end gap-2 rounded-lg border p-1.5 pl-3 shadow-xs transition-[color,box-shadow] focus-within:ring-[3px] dark:bg-input/30"
    >
      <textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        onKeyDown={onKeyDown}
        placeholder={COPY.chatInputPlaceholder}
        aria-label={COPY.chatInputPlaceholder}
        maxLength={MAX_CHARS}
        rows={1}
        disabled={disabled}
        className="placeholder:text-placeholder field-sizing-content max-h-40 min-h-8 min-w-0 flex-1 resize-none bg-transparent py-1.5 text-base outline-none disabled:cursor-not-allowed md:text-sm"
      />
      {text.length > COUNTER_FROM ? (
        <span className="num text-muted-foreground shrink-0 self-center text-xs">
          {text.length}/{MAX_CHARS}
        </span>
      ) : null}
      <Button type="submit" size="icon-sm" aria-label={COPY.chatSend} disabled={!canSend} className="shrink-0">
        <ArrowUpIcon aria-hidden />
      </Button>
    </form>
  );
}
