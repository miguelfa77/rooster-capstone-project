// Signs out a session whose user the API does not know (the (app) layout sends it here).
// Re-checks first, so a link from elsewhere cannot sign out a valid user.
import { redirect } from "next/navigation";
import { signOut } from "@/auth";
import { getCurrentUser } from "@/lib/server-api";

export const dynamic = "force-dynamic";

export async function GET(): Promise<void> {
  if (await getCurrentUser()) redirect("/");
  await signOut({ redirectTo: "/entrar" }); // redirects
}
