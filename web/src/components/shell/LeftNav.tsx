"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useId, type ReactNode } from "react";
import {
  BellIcon,
  BuildingIcon,
  CalculatorIcon,
  ColumnsIcon,
  HouseIcon,
  ListIcon,
  MapIcon,
  type LucideIcon,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { COPY } from "@/lib/copy";
import { cn } from "@/lib/utils";

type NavItem = { label: string; href: string; icon: LucideIcon; available: boolean };

/** In display order. `available: false` = the page does not exist yet. */
const NAV_ITEMS: readonly NavItem[] = [
  { label: COPY.navHome, href: "/", icon: HouseIcon, available: true },
  { label: COPY.navMap, href: "/mapa", icon: MapIcon, available: false },
  { label: COPY.navNeighborhoods, href: "/barrios", icon: BuildingIcon, available: false },
  { label: COPY.navListings, href: "/anuncios", icon: ListIcon, available: false },
  { label: COPY.navCompare, href: "/comparar", icon: ColumnsIcon, available: false },
  { label: COPY.navCalculator, href: "/calculadora", icon: CalculatorIcon, available: false },
  { label: COPY.navWatchlist, href: "/seguimiento", icon: BellIcon, available: false },
];

function isActive(pathname: string, href: string): boolean {
  return href === "/" ? pathname === "/" : pathname === href || pathname.startsWith(`${href}/`);
}

const ITEM = "flex items-center gap-2.5 rounded-lg px-2.5 py-2 text-sm font-medium";

/** Wordmark, primary navigation and the "Conversaciones" slot. */
export function LeftNav({ conversations }: { conversations?: ReactNode }) {
  const pathname = usePathname();
  // The nav renders twice (sidebar and mobile sheet), so ids must be unique per instance.
  const unavailableId = useId();
  const conversationsId = useId();
  return (
    <div className="flex h-full flex-col gap-1 px-3.5 py-4">
      <Link href="/" className="mx-1.5 mb-4 flex items-center gap-2 text-[17px] font-bold">
        <span aria-hidden className="bg-primary size-[22px] rounded-md" />
        {COPY.wordmark}
      </Link>
      <span id={unavailableId} hidden>
        {COPY.navUnavailable}
      </span>
      <nav aria-label={COPY.navLabel}>
        <ul className="flex flex-col gap-0.5">
          {NAV_ITEMS.map(({ label, href, icon: Icon, available }) => (
            <li key={href}>
              {available ? (
                <Link
                  href={href}
                  aria-current={isActive(pathname, href) ? "page" : undefined}
                  className={cn(
                    ITEM,
                    "text-muted-foreground hover:bg-accent hover:text-foreground",
                    "aria-[current=page]:bg-accent aria-[current=page]:text-foreground",
                  )}
                >
                  <Icon aria-hidden className="size-4" />
                  {label}
                </Link>
              ) : (
                // Not focusable and not navigable; announced as a disabled link with a description.
                <span
                  role="link"
                  aria-disabled="true"
                  aria-describedby={unavailableId}
                  className={cn(ITEM, "text-muted-foreground cursor-not-allowed")}
                >
                  <Icon aria-hidden className="size-4 opacity-70" />
                  {label}
                  <Badge variant="secondary" className="text-muted-foreground ml-auto text-[10px]">
                    {COPY.navSoon}
                  </Badge>
                </span>
              )}
            </li>
          ))}
        </ul>
      </nav>
      <section aria-labelledby={conversationsId} className="flex min-h-0 flex-col">
        <h2
          id={conversationsId}
          className="text-muted-foreground mx-2.5 mt-4 mb-1 text-[10.5px] font-semibold tracking-[0.08em] uppercase"
        >
          {COPY.navConversations}
        </h2>
        {conversations}
      </section>
    </div>
  );
}
