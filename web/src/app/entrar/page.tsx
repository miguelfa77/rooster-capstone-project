import { signIn } from "@/auth";
import { Button } from "@/components/ui/button";
import { safeNext } from "@/lib/auth-routing";
import { COPY } from "@/lib/copy";

type Param = string | string[] | undefined;
type Props = { searchParams: Promise<{ next?: Param; callbackUrl?: Param; error?: Param }> };

async function continueWithGoogle(formData: FormData) {
  "use server";
  await signIn("google", { redirectTo: safeNext(formData.get("next")) });
}

export default async function EntrarPage({ searchParams }: Props) {
  // `callbackUrl` is what Auth.js appends when it sends people here itself; `next` wins when both are safe.
  const { next, callbackUrl, error } = await searchParams;
  const target = [next, callbackUrl].map(safeNext).find((path) => path !== "/") ?? "/";
  return (
    <main className="mx-auto flex min-h-svh max-w-sm flex-col items-center justify-center gap-6 p-6 text-center">
      <div className="flex flex-col gap-2">
        <h1 className="text-3xl font-semibold tracking-tight">{COPY.wordmark}</h1>
        <p className="text-muted-foreground">{COPY.tagline}</p>
      </div>
      <form action={continueWithGoogle} className="w-full">
        <input type="hidden" name="next" value={target} />
        <Button type="submit" size="lg" className="w-full">
          {COPY.signInWithGoogle}
        </Button>
      </form>
      {/* Any Auth.js error (pages.error is this page) or our own ?error=servicio. */}
      {error !== undefined && (
        <p role="alert" className="text-destructive text-sm">
          {COPY.signInServiceError}
        </p>
      )}
    </main>
  );
}
