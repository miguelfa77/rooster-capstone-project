import { COPY } from "@/lib/copy";

/** Clickable pills for the latest assistant message; a muted plain line otherwise. */
export default function FollowUps({
  items,
  interactive,
  onPick,
}: {
  items: string[];
  interactive: boolean;
  onPick: (text: string) => void;
}) {
  if (items.length === 0) return null;
  if (!interactive) {
    return (
      <p className="text-muted-foreground text-xs [overflow-wrap:anywhere]">
        {COPY.chatFollowUpsLabel} {items.join(" · ")}
      </p>
    );
  }
  return (
    <div className="flex flex-wrap gap-1.5">
      {items.map((item, i) => (
        <button
          key={i}
          type="button"
          onClick={() => onPick(item)}
          // Tinted Rooster red (mockup .chip): dark red text in light mode, light red in dark mode.
          className="bg-primary/10 hover:bg-primary/15 focus-visible:ring-ring/50 max-w-full rounded-2xl px-3 py-1.5 text-left text-xs font-semibold text-[#cd2b31] outline-none [overflow-wrap:anywhere] focus-visible:ring-[3px] dark:bg-primary/15 dark:text-[#ff8589] dark:hover:bg-primary/20"
        >
          {item}
        </button>
      ))}
    </div>
  );
}
