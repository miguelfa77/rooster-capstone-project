import Link from "next/link";
import { COPY } from "@/lib/copy";
import { getCurrentUser } from "@/lib/server-api";

function firstName(name: string | null | undefined): string | null {
  return name?.trim().split(/\s+/)[0] || null;
}

// Inicio, phase-1 version. Task 5 swaps the pill links for startChat(text) and fills the
// recent-conversations list.
export default async function HomePage() {
  const user = await getCurrentUser();
  const name = firstName(user?.name);
  return (
    <div className="mx-auto flex w-full max-w-3xl flex-col gap-8 px-4 py-8 sm:px-6 sm:py-12">
      <div className="flex flex-col gap-2">
        <h1 className="text-2xl font-semibold tracking-tight sm:text-3xl">
          {name ? COPY.homeGreeting.replace("{name}", name) : COPY.homeGreetingNoName}
        </h1>
        <p className="text-muted-foreground">{COPY.homePrompt}</p>
      </div>
      <section aria-labelledby="home-examples">
        <h2 id="home-examples" className="sr-only">
          {COPY.homeExamplesLabel}
        </h2>
        <ul className="flex flex-wrap gap-2">
          {COPY.chatExamples.map((text) => (
            <li key={text} className="max-w-full">
              <Link
                href={`/chat?q=${encodeURIComponent(text)}`}
                className="bg-card hover:bg-accent focus-visible:ring-ring/50 block rounded-full border px-3.5 py-2 text-sm outline-none focus-visible:ring-[3px]"
              >
                {text}
              </Link>
            </li>
          ))}
        </ul>
      </section>
      <section aria-labelledby="home-recent" className="flex flex-col gap-3">
        <h2 id="home-recent" className="text-base font-semibold">
          {COPY.homeRecentConversations}
        </h2>
        <p className="text-muted-foreground text-sm">{COPY.homeRecentEmpty}</p>
      </section>
    </div>
  );
}
