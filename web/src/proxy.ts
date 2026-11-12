// Next 16 request interception (the renamed `middleware`): every page needs a session
// except the public ones; signed-out visitors go to /entrar?next=<path>.
// /api/rooster/* answers 401 itself, so it is excluded from the matcher.
import { NextResponse } from "next/server";
import { auth } from "@/auth";
import { isPublicPath, signInPath } from "@/lib/auth-routing";

export const proxy = auth((request) => {
  const { pathname, search, origin } = request.nextUrl;
  if (request.auth?.user?.id || isPublicPath(pathname)) return NextResponse.next();
  return NextResponse.redirect(new URL(signInPath(pathname, search), origin));
});

export const config = {
  matcher: ["/((?!api/auth|api/rooster|_next/static|_next/image|favicon\\.ico).*)"],
};
