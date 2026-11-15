"use client";

// The composer's text lives above the chat surfaces (sidebar, full screen, sheet), so switching
// surface keeps what the user was typing.
import { createContext, useContext, useMemo, useState, type ReactNode } from "react";

type Draft = readonly [string, (text: string) => void];

const ComposerDraftContext = createContext<Draft | null>(null);

export function ComposerDraftProvider({ children }: { children: ReactNode }) {
  const [text, setText] = useState("");
  const value = useMemo<Draft>(() => [text, setText], [text]);
  return <ComposerDraftContext.Provider value={value}>{children}</ComposerDraftContext.Provider>;
}

/** The shared composer text; local state when there is no provider (a composer on its own). */
export function useComposerText(): Draft {
  const shared = useContext(ComposerDraftContext);
  const local = useState("");
  return shared ?? local;
}
