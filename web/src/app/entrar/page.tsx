import { signIn } from "@/auth";
import { Button } from "@/components/ui/button";
import { safeNext } from "@/lib/auth-routing";
import { COPY } from "@/lib/copy";

type Props = { searchParams: Promise<{ next?: string | string[]; error?: string | string[] }> };

async function continueWithGoogle(formData: FormData) {
  "use server";
  await signIn("google", { redirectTo: safeNext(formData.get("next")) });
}

export default async function EntrarPage({ searchParams }: Props) {
  const { next, error } = await searchParams;
  return (
    <main className="mx-auto flex min-h-svh max-w-sm flex-col items-center justify-center gap-6 p-6 text-center">
      <div className="flex flex-col gap-2">
        <h1 className="text-3xl font-semibold tracking-tight">{COPY.wordmark}</h1>
        <p className="text-muted-foreground">{COPY.tagline}</p>
      </div>
      <form action={continueWithGoogle} className="w-full">
        <input type="hidden" name="next" value={safeNext(next)} />
        <Button type="submit" size="lg" className="w-full">
          {COPY.signInWithGoogle}
        </Button>
      </form>
      {error === "servicio" && (
        <p role="alert" className="text-destructive text-sm">
          {COPY.signInServiceError}
        </p>
      )}
    </main>
  );
}
