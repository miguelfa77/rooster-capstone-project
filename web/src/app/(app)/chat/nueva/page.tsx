import { FullChat } from "@/components/chat/FullChat";

/** A new conversation, full screen; the URL becomes /chat/<id> once the first message creates it. */
export default function NewChatPage() {
  return <FullChat id={null} />;
}
