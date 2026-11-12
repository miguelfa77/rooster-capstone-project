"use client";

import { useState } from "react";
import { COPY } from "@/lib/copy";
import styles from "./ThinkingBox.module.css";

/** Live-turn reasoning: open while thinking, collapses once prose starts streaming. */
export default function ThinkingBox({ thoughts, hasText }: { thoughts: string[]; hasText: boolean }) {
  const [open, setOpen] = useState(!hasText);
  const [prevHasText, setPrevHasText] = useState(hasText);
  // Collapse on the first text_delta (state adjusted during render, not in an effect).
  if (hasText !== prevHasText) {
    setPrevHasText(hasText);
    if (hasText) setOpen(false);
  }
  const latest = thoughts.at(-1);

  return (
    <div className={styles.box}>
      <button type="button" className={styles.toggle} aria-expanded={open} onClick={() => setOpen((o) => !o)}>
        <span className={styles.label}>{COPY.statusThinking}</span>
        {latest ? <span className={styles.latest}>{latest}</span> : null}
      </button>
      {open && thoughts.length > 0 ? (
        <ol className={styles.list}>
          {thoughts.map((t, i) => (
            <li key={i}>{t}</li>
          ))}
        </ol>
      ) : null}
    </div>
  );
}
