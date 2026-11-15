import Primitive from "@/components/primitives/Primitive";
import type { AssistantMessage as AssistantMessageT, PrimitivePayload } from "@/lib/types";
import { CompactBlock } from "./CompactBlock";
import { inlineInCompact } from "./compact";
import FallbackCard from "./FallbackCard";
import FollowUps from "./FollowUps";

export type Density = "compact" | "full";

/**
 * Wide content (tables, charts, maps) scrolls inside its own block, never the page. In compact
 * density the inline tables drop their wide minimum column widths to fit the column.
 */
const BLOCK = "min-w-0 max-w-full overflow-x-auto";

/**
 * Full density: every block inline. Compact: text, KPIs and small tables inline, the rest as cards.
 * `pos` is the message position, which with the block index keys the open block dialog.
 */
export function Blocks({ items, density, pos }: { items: PrimitivePayload[]; density: Density; pos: number }) {
  return (
    <>
      {items.map((p, i) => {
        const inline = density === "full" || inlineInCompact(p);
        return (
          <div key={i} className={BLOCK} data-testid="message-block">
            {inline ? (
              <Primitive p={p} compact={density === "compact"} />
            ) : (
              <CompactBlock p={p} blockKey={`${pos}:${i}`} />
            )}
          </div>
        );
      })}
    </>
  );
}

export const ASSISTANT = "flex min-w-0 flex-col gap-3";

/** A stored assistant message: fallback card, else blocks, else (legacy) summary + payloads. */
export default function AssistantMessage({
  message,
  interactive,
  onFollowUp,
  density,
  pos,
}: {
  message: AssistantMessageT;
  interactive: boolean;
  onFollowUp: (text: string) => void;
  density: Density;
  pos: number;
}) {
  const fallback = message.graceful_fallback;
  let body;
  if (fallback) {
    body = <FallbackCard reason={fallback.reason} suggestion={fallback.suggestion} />;
  } else if (message.blocks.length > 0) {
    body = <Blocks items={message.blocks} density={density} pos={pos} />;
  } else {
    // Messages stored before `blocks` existed (spec §5.7).
    body = (
      <Blocks
        items={[
          ...(message.summary ? [{ kind: "text" as const, content: message.summary, role: "analysis" as const }] : []),
          ...message.primitive_payloads,
        ]}
        density={density}
        pos={pos}
      />
    );
  }
  return (
    <div className={ASSISTANT}>
      {body}
      <FollowUps items={message.follow_ups} interactive={interactive} onPick={onFollowUp} />
    </div>
  );
}
