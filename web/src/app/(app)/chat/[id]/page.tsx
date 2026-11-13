import { FullChat } from "@/components/chat/FullChat";

/** One conversation, full screen. */
export default async function ChatPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  return <FullChat id={id} />;
}
