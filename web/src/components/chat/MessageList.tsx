"use client";

import { useEffect, useRef } from "react";
import type { LiveTurn } from "@/lib/chat-state";
import { COPY } from "@/lib/copy";
import type { Message } from "@/lib/types";
import { cn } from "@/lib/utils";
import AssistantMessage, { ASSISTANT, Blocks, type Density } from "./AssistantMessage";
import FallbackCard from "./FallbackCard";
import FollowUps from "./FollowUps";
import ThinkingBox from "./ThinkingBox";

/** Within this many px of the bottom counts as "following" the conversation. */
const STICK_PX = 80;

function UserBubble({ text }: { text: string }) {
  return (
    <div className="bg-foreground text-background dark:bg-input dark:text-foreground max-w-[85%] self-end rounded-xl rounded-br-sm px-3 py-2 whitespace-pre-wrap [overflow-wrap:anywhere]">
      {text}
    </div>
  );
}

function LiveTurnView({ live, onFollowUp, density }: { live: LiveTurn; onFollowUp: (text: string) => void; density: Density }) {
  const streaming = live.status === "streaming";
  const blocks = [
    ...(live.text ? [{ kind: "text" as const, content: live.text, role: "analysis" as const }] : []),
    ...live.primitives,
  ];
  return (
    <div className={ASSISTANT}>
      {streaming || live.thoughts.length > 0 ? <ThinkingBox thoughts={live.thoughts} hasText={live.text !== ""} /> : null}
      <Blocks items={blocks} density={density} />
      {live.error ? <FallbackCard reason={live.error.reason} suggestion={live.error.suggestion} /> : null}
      <FollowUps items={live.followUps} interactive={!streaming} onPick={onFollowUp} />
    </div>
  );
}

/** History plus the live turn, in its own scroll container that follows the bottom unless the reader scrolled up. */
export default function MessageList({
  messages,
  live,
  onFollowUp,
  density,
}: {
  messages: Message[];
  live: LiveTurn | null;
  onFollowUp: (text: string) => void;
  density: Density;
}) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const contentRef = useRef<HTMLElement>(null);
  const followRef = useRef(true);

  // Reaching the bottom resumes auto-scroll; only scrolling up stops it. Our own scroll can land
  // short of a bottom that grew meanwhile (maps, plots), which must not count as leaving it.
  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    let lastTop = el.scrollTop;
    const onScroll = () => {
      const top = el.scrollTop;
      if (top + el.clientHeight >= el.scrollHeight - STICK_PX) followRef.current = true;
      else if (top < lastTop) followRef.current = false;
      lastTop = top;
    };
    el.addEventListener("scroll", onScroll, { passive: true });
    return () => el.removeEventListener("scroll", onScroll);
  }, []);

  // Maps, plots and the stored turn lay out after render; keep a reader at the bottom there as the list grows.
  useEffect(() => {
    const content = contentRef.current;
    if (!content || typeof ResizeObserver === "undefined") return;
    const observer = new ResizeObserver(() => {
      const el = scrollRef.current;
      if (el && followRef.current) el.scrollTop = el.scrollHeight;
    });
    observer.observe(content);
    return () => observer.disconnect();
  }, []);

  const prevLiveRef = useRef<LiveTurn | null>(null);
  useEffect(() => {
    const prev = prevLiveRef.current;
    prevLiveRef.current = live;
    // A newly started turn always brings the reader back to the bottom.
    if (live?.status === "streaming" && prev?.status !== "streaming") followRef.current = true;
    const el = scrollRef.current;
    if (el && followRef.current) el.scrollTop = el.scrollHeight;
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
    <div ref={scrollRef} data-testid="chat-scroll" className="min-h-0 flex-1 overflow-y-auto overscroll-contain">
      <section
        ref={contentRef}
        aria-label={COPY.chatMessagesLabel}
        className={cn(
          "flex min-w-0 flex-col",
          density === "full" ? "mx-auto w-full max-w-3xl gap-5 px-4 py-6 sm:px-6" : "gap-4 px-3.5 py-4 text-sm",
        )}
      >
        {messages.map((m, i) =>
          m.role === "user" ? (
            <UserBubble key={i} text={m.content} />
          ) : (
            <AssistantMessage
              key={i}
              message={m}
              interactive={i === latestAssistant}
              onFollowUp={onFollowUp}
              density={density}
            />
          ),
        )}
        {live ? (
          <>
            {live.userText ? <UserBubble text={live.userText} /> : null}
            <LiveTurnView live={live} onFollowUp={onFollowUp} density={density} />
          </>
        ) : null}
      </section>
    </div>
  );
}
