import { ConversationList } from "@/components/chat/ConversationList";
import { COPY } from "@/lib/copy";

/** All of the user's conversations, with rename and delete. */
export default function ChatListPage() {
  return (
    <div className="mx-auto flex w-full max-w-3xl flex-col gap-6 px-4 py-8 sm:px-6 sm:py-12">
      <h1 className="text-2xl font-semibold tracking-tight">{COPY.conversationsTitle}</h1>
      <ConversationList variant="page" />
    </div>
  );
}
