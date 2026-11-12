"use client";

import { ThemeProvider as NextThemesProvider } from "next-themes";

/** Light/dark via a `class` on <html>; follows the OS setting until the user picks one (remembered). */
export function ThemeProvider({ children, ...props }: React.ComponentProps<typeof NextThemesProvider>) {
  return (
    <NextThemesProvider attribute="class" defaultTheme="system" enableSystem disableTransitionOnChange {...props}>
      {children}
    </NextThemesProvider>
  );
}
