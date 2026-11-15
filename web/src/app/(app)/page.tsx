import { ConversationList } from "@/components/chat/ConversationList";
import { ExamplePrompts } from "@/components/chat/ExamplePrompts";
import { COPY } from "@/lib/copy";
import { getCurrentUser } from "@/lib/server-api";

function firstName(name: string | null | undefined): string | null {
  return name?.trim().split(/\s+/)[0] || null;
}

// Inicio: greeting, example questions (each starts a new conversation) and recent conversations.
export default async function HomePage() {
  const user = await getCurrentUser();
  const name = firstName(user?.name);
  return (
    // Left-aligned with the top bar's horizontal padding (px-3 / sm:px-6).
    <div className="flex w-full max-w-3xl flex-col gap-8 px-3 py-8 sm:px-6 sm:py-12">
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
        <ExamplePrompts items={COPY.chatExamples} />
      </section>
      <section aria-labelledby="home-recent" className="flex flex-col gap-3">
        <h2 id="home-recent" className="text-base font-semibold">
          {COPY.homeRecentConversations}
        </h2>
        <ConversationList variant="recent" />
      </section>
    </div>
  );
}
