import { Button } from "@/components/ui/button";
import { COPY } from "@/lib/copy";

// Temporary page proving the theme tokens and shadcn/ui; replaced by the app shell in Task 4.
export default function HomePage() {
  return (
    <main className="mx-auto flex max-w-xl flex-col gap-4 p-6">
      <h1 className="text-2xl font-semibold">{COPY.wordmark}</h1>
      <p className="text-muted-foreground">{COPY.tagline}</p>
      <div className="bg-card rounded-lg border p-4">
        <span className="num text-xl">5,8 %</span>{" "}
        <span className="num text-up">+0,4</span> <span className="num text-down">−1,2</span>
      </div>
      <div>
        <Button>{COPY.navChat}</Button>
      </div>
    </main>
  );
}
