"use client";

import { createContext, useCallback, useContext, useMemo, useState, type ReactNode } from "react";
import Primitive from "@/components/primitives/Primitive";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import type { PrimitivePayload } from "@/lib/types";

/**
 * Which block's dialog is open, as `"<message position>:<index among the non-text blocks>"`. It lives above the
 * live-turn/stored-message boundary (a finished turn is swapped for its stored message, which
 * remounts its blocks), so an open dialog survives that swap.
 */
const OpenBlockContext = createContext<{ openKey: string | null; setOpenKey(key: string | null): void } | null>(null);

export function OpenBlockProvider({ children }: { children: ReactNode }) {
  const [openKey, setOpenKey] = useState<string | null>(null);
  const value = useMemo(() => ({ openKey, setOpenKey }), [openKey]);
  return <OpenBlockContext.Provider value={value}>{children}</OpenBlockContext.Provider>;
}

/** `blockKey` ties the dialog to the shared open state; without a provider it is uncontrolled. */
function useBlockOpen(blockKey: string | undefined) {
  const ctx = useContext(OpenBlockContext);
  const [local, setLocal] = useState(false);
  const onOpenChange = useCallback(
    (open: boolean) => {
      if (!ctx || blockKey === undefined) setLocal(open);
      else if (open) ctx.setOpenKey(blockKey);
      else if (ctx.openKey === blockKey) ctx.setOpenKey(null);
    },
    [ctx, blockKey],
  );
  const open = ctx && blockKey !== undefined ? ctx.openKey === blockKey : local;
  return { open, onOpenChange };
}

/** A large dialog with the full, interactive block; `trigger` opens it and gets focus back on close. */
export function BlockDialog({
  p,
  title,
  trigger,
  blockKey,
}: {
  p: PrimitivePayload;
  title: string;
  trigger: ReactNode;
  blockKey?: string;
}) {
  const { open, onOpenChange } = useBlockOpen(blockKey);
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogTrigger asChild>{trigger}</DialogTrigger>
      <DialogContent
        aria-describedby={undefined}
        className="flex max-h-[90svh] flex-col gap-4 p-4 sm:max-w-[min(64rem,calc(100%-2rem))] sm:p-6"
      >
        <DialogHeader className="pr-8 text-left">
          <DialogTitle className="text-base leading-snug">{title}</DialogTitle>
        </DialogHeader>
        <div className="min-h-0 min-w-0 flex-1 overflow-auto">
          <Primitive p={p} />
        </div>
      </DialogContent>
    </Dialog>
  );
}
