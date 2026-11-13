"use client";

import type { ReactNode } from "react";
import Primitive from "@/components/primitives/Primitive";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import type { PrimitivePayload } from "@/lib/types";

/** A large dialog with the full, interactive block; `trigger` opens it and gets focus back on close. */
export function BlockDialog({ p, title, trigger }: { p: PrimitivePayload; title: string; trigger: ReactNode }) {
  return (
    <Dialog>
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
