import { signIn } from "@/auth";
import { Button } from "@/components/ui/button";
import { COPY } from "@/lib/copy";

async function tryAnotherAccount() {
  "use server";
  // Google remembers the last account; ask it to show the account picker again.
  await signIn("google", { redirectTo: "/" }, { prompt: "select_account" });
}

export default function InvitacionPage() {
  return (
    <main className="mx-auto flex min-h-svh max-w-sm flex-col items-center justify-center gap-6 p-6 text-center">
      <div className="flex flex-col gap-2">
        <h1 className="text-2xl font-semibold tracking-tight">{COPY.inviteTitle}</h1>
        <p className="text-muted-foreground">{COPY.inviteHowToRequest}</p>
      </div>
      <form action={tryAnotherAccount} className="w-full">
        <Button type="submit" variant="outline" size="lg" className="w-full">
          {COPY.inviteTryAnotherAccount}
        </Button>
      </form>
    </main>
  );
}
