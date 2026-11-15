// Pure helpers shared by the request-interception proxy and the sign-in pages.

/** Same-origin relative path or "/". Rejects "//host" and "/\host" (browsers read both as another origin). */
export function safeNext(next: unknown): string {
  if (typeof next !== "string" || !next.startsWith("/") || next.startsWith("//") || next.includes("\\")) return "/";
  return next;
}

/** Pages reachable without a session. Next static assets are excluded by the proxy matcher. */
export function isPublicPath(pathname: string): boolean {
  return (
    pathname === "/entrar" ||
    pathname.startsWith("/entrar/") ||
    pathname.startsWith("/compartir/") ||
    pathname.startsWith("/api/auth/")
  );
}

export function signInPath(pathname: string, search: string): string {
  return `/entrar?${new URLSearchParams({ next: `${pathname}${search}` })}`;
}
