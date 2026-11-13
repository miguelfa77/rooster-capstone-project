"use client";

import { ChevronRightIcon } from "lucide-react";
import { useState } from "react";
import { COPY } from "@/lib/copy";

/** Live-turn reasoning: open while thinking, collapses once prose starts streaming. */
export default function ThinkingBox({ thoughts, hasText }: { thoughts: string[]; hasText: boolean }) {
  const [open, setOpen] = useState(!hasText);
  const [prevHasText, setPrevHasText] = useState(hasText);
  // Collapse on the first text_delta and reopen when a new turn starts without text
  // (state adjusted during render, not in an effect).
  if (hasText !== prevHasText) {
    setPrevHasText(hasText);
    setOpen(!hasText);
  }
  const latest = thoughts.at(-1);

  return (
    <div className="text-muted-foreground text-xs">
      <button
        type="button"
        aria-expanded={open}
        onClick={() => setOpen((o) => !o)}
        className="hover:text-foreground focus-visible:ring-ring/50 -mx-1 flex w-[calc(100%+0.5rem)] min-w-0 items-center gap-1.5 rounded-md px-1 py-0.5 text-left outline-none focus-visible:ring-[3px]"
      >
        <ChevronRightIcon aria-hidden className={`size-3.5 shrink-0 transition-transform ${open ? "rotate-90" : ""}`} />
        <span className="text-foreground shrink-0 font-medium">{COPY.statusThinking}</span>
        {latest ? <span className="min-w-0 truncate">{latest}</span> : null}
      </button>
      {open && thoughts.length > 0 ? (
        <ol className="mt-1 list-decimal space-y-0.5 pl-9 [overflow-wrap:anywhere]">
          {thoughts.map((t, i) => (
            <li key={i}>{t}</li>
          ))}
        </ol>
      ) : null}
    </div>
  );
}
