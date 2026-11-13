// Where the user goes to read a conversation. One place, so the chat panel (Task 6) can
// swap navigation for "open the sidebar".
type Router = { push(href: string): void };

export function showConversation(router: Router, id: string): void {
  router.push(`/chat/${encodeURIComponent(id)}`);
}
