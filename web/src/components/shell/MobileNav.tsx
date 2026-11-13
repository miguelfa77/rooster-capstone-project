"use client";

import { MenuIcon } from "lucide-react";
import { useState, type ReactNode } from "react";
import { Button } from "@/components/ui/button";
import { Sheet, SheetContent, SheetTitle, SheetTrigger } from "@/components/ui/sheet";
import { COPY } from "@/lib/copy";

/** Below 1024 px the left nav lives in a sheet opened from the top bar; following a link closes it. */
export function MobileNav({ children }: { children: ReactNode }) {
  const [open, setOpen] = useState(false);
  return (
    <Sheet open={open} onOpenChange={setOpen}>
      <SheetTrigger asChild>
        <Button variant="ghost" size="icon" className="lg:hidden" aria-label={COPY.navOpenMenu}>
          <MenuIcon aria-hidden />
        </Button>
      </SheetTrigger>
      <SheetContent
        side="left"
        className="bg-card w-[260px] gap-0 p-0"
        aria-describedby={undefined}
        onClick={(e) => {
          if ((e.target as Element).closest("a[href]")) setOpen(false);
        }}
      >
        <SheetTitle className="sr-only">{COPY.navLabel}</SheetTitle>
        {children}
      </SheetContent>
    </Sheet>
  );
}
