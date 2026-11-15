"use client";

import { useCallback, useSyncExternalStore } from "react";

/** The shell's `lg` breakpoint: left nav and chat sidebar columns from here up. */
export const DESKTOP_QUERY = "(min-width: 1024px)";

/**
 * Whether a media query matches. The server (and the hydrating client) assume `serverDefault`;
 * markup that must not flash on the other size should also carry the matching Tailwind class.
 */
export function useMediaQuery(query: string, serverDefault: boolean): boolean {
  const subscribe = useCallback(
    (onChange: () => void) => {
      if (typeof window.matchMedia !== "function") return () => {};
      const mql = window.matchMedia(query);
      mql.addEventListener("change", onChange);
      return () => mql.removeEventListener("change", onChange);
    },
    [query],
  );
  return useSyncExternalStore(
    subscribe,
    () => (typeof window.matchMedia === "function" ? window.matchMedia(query).matches : serverDefault),
    () => serverDefault,
  );
}

export function useIsDesktop(): boolean {
  return useMediaQuery(DESKTOP_QUERY, true);
}
