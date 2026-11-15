"use client";

import { ThemeProvider as NextThemesProvider } from "next-themes";

/** Light/dark via a `class` on <html>: light by default, the user picks dark or "Sistema" (follows the OS); remembered. */
export function ThemeProvider({ children, ...props }: React.ComponentProps<typeof NextThemesProvider>) {
  return (
    <NextThemesProvider attribute="class" defaultTheme="light" enableSystem disableTransitionOnChange {...props}>
      {children}
    </NextThemesProvider>
  );
}
