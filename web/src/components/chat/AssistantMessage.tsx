import Primitive from "@/components/primitives/Primitive";
import TextBlock from "@/components/primitives/TextBlock";
import type { AssistantMessage as AssistantMessageT, PrimitivePayload } from "@/lib/types";
import FallbackCard from "./FallbackCard";
import FollowUps from "./FollowUps";
import styles from "./AssistantMessage.module.css";

export function Blocks({ items }: { items: PrimitivePayload[] }) {
  return (
    <>
      {items.map((p, i) => (
        <div key={i} className={styles.block} data-testid="message-block">
          <Primitive p={p} />
        </div>
      ))}
    </>
  );
}

/** A stored assistant message: fallback card, else blocks, else (legacy) summary + payloads. */
export default function AssistantMessage({
  message,
  interactive,
  onFollowUp,
}: {
  message: AssistantMessageT;
  interactive: boolean;
  onFollowUp: (text: string) => void;
}) {
  const fallback = message.graceful_fallback;
  let body;
  if (fallback) {
    body = <FallbackCard reason={fallback.reason} suggestion={fallback.suggestion} />;
  } else if (message.blocks.length > 0) {
    body = <Blocks items={message.blocks} />;
  } else {
    // Messages stored before `blocks` existed (spec §5.7).
    body = (
      <>
        {message.summary ? (
          <div className={styles.block} data-testid="message-block">
            <TextBlock p={{ kind: "text", content: message.summary, role: "analysis" }} />
          </div>
        ) : null}
        <Blocks items={message.primitive_payloads} />
      </>
    );
  }
  return (
    <div className={styles.assistant}>
      {body}
      <FollowUps items={message.follow_ups} interactive={interactive} onPick={onFollowUp} />
    </div>
  );
}
