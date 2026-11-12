"use client";

import { useEffect, useRef } from "react";
import TextBlock from "@/components/primitives/TextBlock";
import type { LiveTurn } from "@/lib/chat-state";
import { COPY } from "@/lib/copy";
import type { Message } from "@/lib/types";
import AssistantMessage, { Blocks } from "./AssistantMessage";
import FallbackCard from "./FallbackCard";
import FollowUps from "./FollowUps";
import ThinkingBox from "./ThinkingBox";
import styles from "./AssistantMessage.module.css";
import listStyles from "./MessageList.module.css";

/** Within this many px of the bottom counts as "following" the conversation. */
const STICK_PX = 80;

/** The page bottom, so the sticky composer sits below the last message instead of covering it. */
function scrollToBottom() {
  window.scrollTo({ top: document.documentElement.scrollHeight });
}

function UserBubble({ text }: { text: string }) {
  return <div className={listStyles.user}>{text}</div>;
}

function LiveTurnView({ live, onFollowUp }: { live: LiveTurn; onFollowUp: (text: string) => void }) {
  const streaming = live.status === "streaming";
  return (
    <div className={styles.assistant}>
      {streaming || live.thoughts.length > 0 ? <ThinkingBox thoughts={live.thoughts} hasText={live.text !== ""} /> : null}
      {live.text ? (
        <div className={styles.block} data-testid="message-block">
          <TextBlock p={{ kind: "text", content: live.text, role: "analysis" }} />
        </div>
      ) : null}
      <Blocks items={live.primitives} />
      {live.error ? <FallbackCard reason={live.error.reason} suggestion={live.error.suggestion} /> : null}
      <FollowUps items={live.followUps} interactive={!streaming} onPick={onFollowUp} />
    </div>
  );
}

export default function MessageList({
  messages,
  live,
  onFollowUp,
}: {
  messages: Message[];
  live: LiveTurn | null;
  onFollowUp: (text: string) => void;
}) {
  const listRef = useRef<HTMLElement>(null);
  const followRef = useRef(true);

  // Reaching the bottom resumes auto-scroll; only scrolling up stops it. Our own scroll can land
  // short of a bottom that grew meanwhile (maps, plots), which must not count as leaving it.
  useEffect(() => {
    let lastY = window.scrollY;
    const onScroll = () => {
      const y = window.scrollY;
      if (window.innerHeight + y >= document.documentElement.scrollHeight - STICK_PX) followRef.current = true;
      else if (y < lastY) followRef.current = false;
      lastY = y;
    };
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  // Maps, plots and the stored turn lay out after render; keep a reader at the bottom there as the list grows.
  useEffect(() => {
    const list = listRef.current;
    if (!list || typeof ResizeObserver === "undefined") return;
    const observer = new ResizeObserver(() => {
      if (followRef.current) scrollToBottom();
    });
    observer.observe(list);
    return () => observer.disconnect();
  }, []);

  const prevLiveRef = useRef<LiveTurn | null>(null);
  useEffect(() => {
    const prev = prevLiveRef.current;
    prevLiveRef.current = live;
    // A newly started turn always brings the reader back to the bottom.
    if (live?.status === "streaming" && prev?.status !== "streaming") followRef.current = true;
    if (followRef.current) scrollToBottom();
  }, [messages, live]);

  // Only the latest assistant message gets clickable pills; a live turn is always the latest.
  let latestAssistant = -1;
  if (!live) {
    for (let i = messages.length - 1; i >= 0; i--) {
      if (messages[i].role === "assistant") {
        latestAssistant = i;
        break;
      }
    }
  }

  return (
    <section ref={listRef} className={listStyles.list} aria-label={COPY.chatMessagesLabel}>
      {messages.map((m, i) =>
        m.role === "user" ? (
          <UserBubble key={i} text={m.content} />
        ) : (
          <AssistantMessage key={i} message={m} interactive={i === latestAssistant} onFollowUp={onFollowUp} />
        ),
      )}
      {live ? (
        <>
          {live.userText ? <UserBubble text={live.userText} /> : null}
          <LiveTurnView live={live} onFollowUp={onFollowUp} />
        </>
      ) : null}
    </section>
  );
}
