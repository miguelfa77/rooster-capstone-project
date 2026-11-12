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

function UserBubble({ text }: { text: string }) {
  return <div className={listStyles.user}>{text}</div>;
}

function LiveTurnView({ live, onFollowUp }: { live: LiveTurn; onFollowUp: (text: string) => void }) {
  const streaming = live.status === "streaming";
  return (
    <div className={styles.assistant}>
      {streaming || live.thoughts.length > 0 ? <ThinkingBox thoughts={live.thoughts} hasText={live.text !== ""} /> : null}
      {live.text ? (
        <div className={styles.block}>
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
  const endRef = useRef<HTMLDivElement>(null);
  const followRef = useRef(true);

  // Track whether the reader is at the bottom of the page; scrolling up stops auto-scroll.
  useEffect(() => {
    const onScroll = () => {
      const doc = document.documentElement;
      followRef.current = window.innerHeight + window.scrollY >= doc.scrollHeight - STICK_PX;
    };
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  useEffect(() => {
    if (followRef.current) endRef.current?.scrollIntoView?.({ block: "end" });
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
    <section className={listStyles.list} aria-label={COPY.chatMessagesLabel}>
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
      <div ref={endRef} />
    </section>
  );
}
