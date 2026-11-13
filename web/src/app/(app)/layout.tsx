import { redirect } from "next/navigation";
import type { ReactNode } from "react";
import { AppShell } from "@/components/shell/AppShell";
import { Toaster } from "@/components/ui/sonner";
import { getCurrentUser } from "@/lib/server-api";
import { PreferencesProvider } from "@/lib/use-preferences";

// Every signed-in page. The user (with stored preferences) is fetched here on the server;
// a session whose user the API no longer knows is signed out via /salir.
export default async function AppLayout({ children }: { children: ReactNode }) {
  const user = await getCurrentUser();
  if (!user) redirect("/salir");
  return (
    <PreferencesProvider initial={user.preferences}>
      <AppShell user={user}>{children}</AppShell>
      <Toaster />
    </PreferencesProvider>
  );
}
