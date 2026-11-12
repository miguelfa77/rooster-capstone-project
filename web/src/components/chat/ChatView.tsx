"use client";

import { COPY } from "@/lib/copy";
import { useChat } from "@/lib/use-chat";
import Composer from "./Composer";
import MessageList from "./MessageList";
import pillStyles from "./FollowUps.module.css";
import styles from "./ChatView.module.css";

function EmptyChat({ onPick }: { onPick: (text: string) => void }) {
  return (
    <div className={styles.empty} data-testid="chat-empty">
      <h1 className={styles.wordmark}>{COPY.wordmark}</h1>
      <p className={styles.tagline}>{COPY.tagline}</p>
      <div className={pillStyles.pills}>
        {COPY.chatExamples.map((q) => (
          <button key={q} type="button" className={pillStyles.pill} onClick={() => onPick(q)}>
            {q}
          </button>
        ))}
      </div>
    </div>
  );
}

/** The chat page: history, the live turn, and the composer. */
export default function ChatView() {
  const { state, send, reset } = useChat();
  const { messages, live, notice, loading } = state;
  const busy = loading || live?.status === "streaming";
  const isEmpty = !loading && messages.length === 0 && !live;
  const onSend = (text: string) => void send(text);

  return (
    <div className={styles.chat}>
      {isEmpty ? <EmptyChat onPick={onSend} /> : <MessageList messages={messages} live={live} onFollowUp={onSend} />}
      <div className={styles.dock}>
        {notice ? (
          <p className={styles.notice} role="status">
            {notice}
          </p>
        ) : null}
        <Composer disabled={busy} onSend={onSend} onReset={reset} />
      </div>
    </div>
  );
}
