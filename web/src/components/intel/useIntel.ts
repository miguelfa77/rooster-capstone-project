"use client";

import { useEffect, useState } from "react";
import { getIntel } from "@/lib/client";

export type IntelState<T> = { status: "loading" } | { status: "error" } | { status: "ready"; data: T };

/** Fetches one /intel/* resource on mount; each section owns its own state. */
export function useIntel<T>(path: string): IntelState<T> {
  const [state, setState] = useState<IntelState<T>>({ status: "loading" });
  useEffect(() => {
    const ctrl = new AbortController();
    getIntel<T>(path, ctrl.signal).then(
      (data) => setState({ status: "ready", data }),
      (err: unknown) => {
        if (ctrl.signal.aborted) return;
        console.warn(`intel ${path} failed`, err);
        setState({ status: "error" });
      },
    );
    return () => ctrl.abort();
  }, [path]);
  return state;
}
