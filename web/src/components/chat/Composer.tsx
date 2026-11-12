"use client";

import { useState, type KeyboardEvent } from "react";
import { COPY } from "@/lib/copy";
import styles from "./Composer.module.css";

const MAX_CHARS = 2000;
const COUNTER_FROM = 1800;

/** Message input: Enter sends, Shift+Enter inserts a newline. A new `draft` (a refused send) refills it. */
export default function Composer({
  disabled,
  draft = null,
  onSend,
  onReset,
}: {
  disabled: boolean;
  draft?: { text: string } | null;
  onSend: (text: string) => void;
  onReset: () => void;
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
      className={styles.composer}
      onSubmit={(e) => {
        e.preventDefault();
        submit();
      }}
    >
      <textarea
        className={styles.input}
        value={text}
        onChange={(e) => setText(e.target.value)}
        onKeyDown={onKeyDown}
        placeholder={COPY.chatInputPlaceholder}
        aria-label={COPY.chatInputPlaceholder}
        maxLength={MAX_CHARS}
        rows={2}
        disabled={disabled}
      />
      <div className={styles.row}>
        <button type="button" className={styles.secondary} onClick={onReset}>
          {COPY.newConversation}
        </button>
        {text.length > COUNTER_FROM ? (
          <span className={styles.counter}>
            {text.length}/{MAX_CHARS}
          </span>
        ) : null}
        <button type="submit" className={styles.send} disabled={!canSend}>
          {COPY.chatSend}
        </button>
      </div>
    </form>
  );
}
