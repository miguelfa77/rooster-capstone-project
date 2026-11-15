// Auth.js: Google sign-in, invite-only, JWT sessions. Reads AUTH_SECRET,
// AUTH_GOOGLE_ID and AUTH_GOOGLE_SECRET from the environment.
import NextAuth from "next-auth";
import Google from "next-auth/providers/google";
import { jwtCallback, sessionCallback, signInCallback } from "@/lib/auth-callbacks";

export const { auth, handlers, signIn, signOut } = NextAuth({
  providers: [Google],
  session: { strategy: "jwt" },
  pages: { signIn: "/entrar", error: "/entrar" },
  callbacks: {
    signIn: signInCallback,
    jwt: jwtCallback,
    session: sessionCallback,
  },
});

/** The signed-in user's Rooster id, or null when signed out. */
export async function currentUserId(): Promise<string | null> {
  const session = await auth();
  return session?.user?.id ?? null;
}
