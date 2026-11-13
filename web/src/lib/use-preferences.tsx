"use client";

// Per-user UI preferences: initial value from the server, optimistic updates, PATCH in the
// background, rollback (with a toast) if the PATCH fails. `theme` also drives next-themes.
import { useTheme } from "next-themes";
import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { toast } from "sonner";
import { patchPreferences, SignedOutError } from "./client";
import { COPY } from "./copy";
import type { Preferences } from "./types";

type Value = { prefs: Preferences; set(p: Partial<Preferences>): void };

const PreferencesContext = createContext<Value | null>(null);

export function PreferencesProvider({ initial, children }: { initial: Preferences; children: ReactNode }) {
  const [prefs, setPrefs] = useState<Preferences>(initial);
  const current = useRef<Preferences>(initial);
  const { setTheme } = useTheme();

  // Apply the stored theme once on load; without one, next-themes keeps following the system.
  // (setTheme's identity changes with the theme, so guard rather than rely on deps.)
  const pendingInitialTheme = useRef(initial.theme);
  useEffect(() => {
    const stored = pendingInitialTheme.current;
    pendingInitialTheme.current = null;
    if (stored) setTheme(stored);
  }, [setTheme]);

  const set = useCallback(
    (patch: Partial<Preferences>) => {
      const keys = (Object.keys(patch) as (keyof Preferences)[]).filter((k) => patch[k] !== undefined);
      if (keys.length === 0) return;
      const changes: Preferences = Object.fromEntries(keys.map((k) => [k, patch[k]]));
      const previous: Preferences = Object.fromEntries(keys.map((k) => [k, current.current[k]]));

      current.current = { ...current.current, ...changes };
      setPrefs(current.current);
      if (changes.theme) setTheme(changes.theme);

      patchPreferences(changes).catch((err: unknown) => {
        if (err instanceof SignedOutError) return;
        // Roll back only the keys no later change has overwritten since.
        const restore = keys.filter((k) => current.current[k] === changes[k]);
        if (restore.length === 0) return;
        current.current = { ...current.current, ...Object.fromEntries(restore.map((k) => [k, previous[k]])) };
        setPrefs(current.current);
        if (restore.includes("theme")) setTheme(previous.theme ?? "system");
        toast.error(COPY.preferencesSaveError);
      });
    },
    [setTheme],
  );

  const value = useMemo(() => ({ prefs, set }), [prefs, set]);
  return <PreferencesContext.Provider value={value}>{children}</PreferencesContext.Provider>;
}

export function usePreferences(): Value {
  const value = useContext(PreferencesContext);
  if (!value) throw new Error("usePreferences must be used inside <PreferencesProvider>");
  return value;
}
