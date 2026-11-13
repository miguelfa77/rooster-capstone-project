"use client";

import { LogOutIcon } from "lucide-react";
import { useTheme } from "next-themes";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { COPY } from "@/lib/copy";
import { signOutAction } from "@/lib/session-actions";
import type { Preferences, UserOut } from "@/lib/types";
import { usePreferences } from "@/lib/use-preferences";

type Theme = NonNullable<Preferences["theme"]>;

const THEMES: readonly { value: Theme; label: string }[] = [
  { value: "light", label: COPY.themeLight },
  { value: "dark", label: COPY.themeDark },
  { value: "system", label: COPY.themeSystem },
];

function isTheme(value: unknown): value is Theme {
  return THEMES.some((t) => t.value === value);
}

/** Up to two initials from the name, else the first letter of the email. */
export function initials(user: Pick<UserOut, "name" | "email">): string {
  const words = (user.name ?? "").trim().split(/\s+/).filter(Boolean);
  const letters = words.length > 1 ? [words[0], words[words.length - 1]] : words;
  const out = letters.map((w) => w[0]).join("") || user.email[0] || "?";
  return out.toUpperCase();
}

export function UserMenu({ user }: { user: UserOut }) {
  const { prefs, set } = usePreferences();
  const { theme } = useTheme();
  const current = prefs.theme ?? (isTheme(theme) ? theme : "system");

  return (
    <DropdownMenu>
      <DropdownMenuTrigger
        aria-label={COPY.userMenuLabel}
        className="focus-visible:ring-ring/50 rounded-full outline-none focus-visible:ring-[3px]"
      >
        <Avatar className="size-8">
          {user.image && <AvatarImage src={user.image} alt="" />}
          <AvatarFallback className="bg-foreground text-background dark:bg-input dark:text-foreground text-xs font-semibold">
            {initials(user)}
          </AvatarFallback>
        </Avatar>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-60">
        <DropdownMenuLabel className="flex flex-col gap-0.5">
          {user.name && <span className="truncate">{user.name}</span>}
          <span className="text-muted-foreground truncate text-xs font-normal">{user.email}</span>
        </DropdownMenuLabel>
        <DropdownMenuSeparator />
        <DropdownMenuLabel className="text-muted-foreground text-xs font-normal">{COPY.themeLabel}</DropdownMenuLabel>
        <DropdownMenuRadioGroup
          value={current}
          onValueChange={(value) => {
            if (isTheme(value)) set({ theme: value });
          }}
        >
          {THEMES.map((t) => (
            <DropdownMenuRadioItem key={t.value} value={t.value}>
              {t.label}
            </DropdownMenuRadioItem>
          ))}
        </DropdownMenuRadioGroup>
        <DropdownMenuSeparator />
        <DropdownMenuItem onSelect={() => void signOutAction()}>
          <LogOutIcon aria-hidden />
          {COPY.signOut}
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
