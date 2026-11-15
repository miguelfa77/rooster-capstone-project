# Rooster v4 Phase 1 — Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A signed-in, invite-only Rooster with the new product shell (light/dark, left nav, top bar) and the chat as sidebar / full screen / phone sheet with per-user conversation history.

**Architecture:** The API gains users and makes conversations user-owned; the Next.js server signs people in with Google (Auth.js, JWT sessions), checks an email allowlist, and forwards the user's ID to the API behind the existing secret key. The web app is rebuilt in place under `web/` on Tailwind + shadcn/ui; v3's tested plumbing (proxy, SSE, chat reducer/hook, generated types, primitives, maps) is kept, v3's screens are removed.

**Tech Stack:** FastAPI · SQLAlchemy · PostgreSQL · Next.js 16 (App Router) · React 19 · TypeScript strict · Tailwind CSS v4 · shadcn/ui (Radix) · next-themes · Auth.js (`next-auth`, current stable for Next 16) · lucide-react · Vitest · Playwright.

**Spec:** `docs/superpowers/specs/2026-11-12-v4-product-design.md` (§5.1–5.3, §5.8, §5.10, §6 phase 1, §7, §7b). Mockup: `docs/design/v4-mockups/ticker.html`.

## Global Constraints

- UI copy Spanish, kept in `web/src/lib/copy.ts`; code, docs, commits English. Commits carry **no AI attribution** (no `Co-Authored-By`, session links, "Generated with").
- Light mode is the default; dark is a toggle; follows the system setting until the user chooses; remembered per user.
- Rooster red `#e5484d` only for primary actions/highlights; green/red for up/down; amber for warnings. Text font Inter; numbers in JetBrains Mono with tabular figures (`next/font`).
- The browser never sees `ROOSTER_API_KEY`, `AUTH_SECRET` or Google secrets and never calls FastAPI directly. No host-specific APIs.
- Sign-in: Google only. Invite-only via server env `ROOSTER_ALLOWED_EMAILS` (comma-separated, case-insensitive, whitespace-trimmed). Empty/unset allowlist ⇒ nobody can sign in.
- Everything requires sign-in except `/entrar`, `/compartir/*` and Auth.js's own routes.
- Identity to the API: header `X-Rooster-User: <user uuid>` next to `X-Rooster-Key`. A user-owned resource requested by another user ⇒ 404.
- No agent behavior change. `python3 -m pytest -q` and all web checks stay green after every task.
- Next.js 16 differs from older versions (e.g. middleware naming): consult `web/node_modules/next/dist/docs/` before using a Next API.
- If the shadcn CLI cannot reach its registry from this machine, add the same components by hand from Radix packages on npm, keeping shadcn's file layout (`web/src/components/ui/*`, `web/src/lib/utils.ts` with `cn`).
- Phone width 390 px: no horizontal page scroll.

## Review Focus

1. **Allowlist edge cases** — `" Ana@Fund.com "` in the env matches `ana@fund.com`; unset/empty env rejects everyone; a rejected Google account lands on the invite-only page, not an error (Task 3).
2. **Session missing or expired mid-use** — proxy answers 401 `{"detail":"signed_out"}`; the client sends the person to `/entrar` instead of showing a broken chat (Tasks 3, 5).
3. **Someone else's conversation ID** (typed URL, stale localStorage after switching account) — API 404; the web clears it and starts fresh, never shows another user's messages (Tasks 1, 5).
4. **Deleting or renaming the conversation that is open** — deleting resets the chat to a new empty conversation everywhere it is shown; renaming updates nav and header at once (Task 5).
5. **A turn in progress while switching sidebar ⇄ full screen ⇄ another page** — the stream continues and the same live turn is shown in the new place (Task 6).

---

## File Structure

```
sql/migrate_app_users.sql                 NEW  app.users; conversations.user_id/title
api/store.py                              MOD  users, per-user conversations, preferences
api/models.py, api/main.py                MOD  UserOut, ConversationSummary, routes, require_user
api/openapi.json, web/src/lib/api-types.ts REGEN
web/
  postcss/tailwind setup, components.json, src/app/globals.css (theme tokens)
  src/components/ui/*                     shadcn components
  src/lib/utils.ts (cn), theme-provider.tsx
  src/auth.ts, src/lib/allowlist.ts, proxy/middleware file, src/app/api/auth/[...nextauth]/route.ts
  src/app/entrar/page.tsx, src/app/entrar/invitacion/page.tsx
  src/app/(app)/layout.tsx                signed-in shell
  src/app/(app)/page.tsx                  Inicio (phase-1 version)
  src/app/(app)/chat/page.tsx, chat/[id]/page.tsx
  src/components/shell/{AppShell,LeftNav,TopBarV4,UserMenu,MobileNav}.tsx
  src/components/chat/*                   rebuilt on Tailwind (ChatPanel, ChatSidebar, ChatSheet, ConversationList, …)
  src/lib/{client.ts,use-chat.ts,chat-state.ts,chat-context.tsx,copy.ts,types.ts}  MOD
  src/lib/{use-conversations.ts,use-preferences.ts}  NEW
  tests/unit/*, tests/e2e/*, tests/fake-api/server.mjs  MOD/NEW
docker-compose.yml, .env.example, README.md, .claude/CLAUDE.md  MOD
REMOVED: web/src/app/intel/, web/src/components/TopBar.tsx(+css,test), web/tests/e2e/intel.spec.ts, chat *.module.css
KEPT for Phase 2: web/src/components/{map,intel,primitives}/ (restyled by theme tokens only)
```

---

### Task 1: API — users, user-owned conversations, preferences

**Files:** Create `sql/migrate_app_users.sql`, `tests/api/test_users.py`. Modify `api/store.py`, `api/models.py`, `api/main.py`, `api/export_openapi.py` (if needed), `api/openapi.json`, `tests/api/test_main.py`, `tests/api/test_store.py`, `tests/eval/run_eval.py` (API mode must create a user and send the header), `README.md` (migration step only).

**Interfaces — Produces:**
- SQL (idempotent): `app.users(id uuid pk, email text not null unique /* stored lowercased */, name text, image text, preferences jsonb not null default '{}', created_at timestamptz default now(), last_seen_at timestamptz default now())`; `app.conversations` gains `user_id uuid references app.users(id) on delete cascade` and `title text`; rows with `user_id IS NULL` are deleted, then `user_id` is set `NOT NULL`; index `(user_id, updated_at desc)`.
- Store: `upsert_user(email: str, name: str | None, image: str | None) -> dict` (id, email, name, image, preferences; lowercases email; updates name/image/last_seen_at); `get_user(user_id) -> dict | None`; `update_preferences(user_id, patch: dict) -> dict` (shallow merge); `create(user_id) -> uuid`; `load(conversation_id, user_id) -> TurnState | None` (None when not owned); `list_conversations(user_id) -> list[dict]` (id, title, updated_at; newest first; max 100); `rename(conversation_id, user_id, title) -> bool`; `delete(conversation_id, user_id) -> bool`; `save` sets `title` when NULL to the first user message's text, whitespace-collapsed and cut to 60 characters.
- HTTP (all need `X-Rooster-Key`; all but `PUT /users` need `X-Rooster-User`, else 401 `{"detail":"user_required"}`; unknown user id ⇒ 401 too):
  - `PUT /users` body `{email, name?, image?}` → 200 `UserOut {id, email, name, image, preferences}`.
  - `GET /me` → `UserOut`. `PATCH /me/preferences` body `Preferences {theme?: "light"|"dark"|"system", chat_open?: bool, ticker_visible?: bool}` (unknown keys rejected 422) → merged `Preferences`.
  - `GET /conversations` → `list[ConversationSummary {id, title: str | None, updated_at}]`.
  - `POST /conversations` → 201 `{id}` · `GET /conversations/{id}` → `ConversationOut` now also with `title` · `PATCH /conversations/{id}` body `{title}` (trimmed, 1–120 chars, else 422) → `ConversationSummary` · `DELETE /conversations/{id}` → 204 · `POST /conversations/{id}/messages` unchanged apart from ownership.
  - Not owned / not found ⇒ 404 on every conversation route. `/intel/*` and `/health` unchanged (no user needed).
- OpenAPI contains `UserOut`, `Preferences`, `ConversationSummary`; `api/openapi.json` regenerated.

- [ ] **Step 1: Failing tests.** `tests/api/test_users.py` + additions (fake store extended to match): `put_users_upserts_and_lowercases` · `me_requires_user_header_401` · `unknown_user_401` · `preferences_merge_and_reject_unknown_key_422` · `conversations_listed_newest_first_only_mine` · `other_users_conversation_is_404` (GET, PATCH, DELETE, POST messages) · `rename_trims_and_validates_length` · `delete_then_404` · `title_set_from_first_message_60_chars` · `intel_needs_no_user`. Real-DB tests in `tests/api/test_store.py` for each store method and `migration_deletes_ownerless_conversations_and_is_idempotent`.
- [ ] **Step 2:** `python3 -m pytest tests/api -q` → FAIL.
- [ ] **Step 3: Implement** (dependency `require_user(x_rooster_user) -> uuid.UUID` in `api/main.py`).
- [ ] **Step 4:** `python3 -m pytest -q` → PASS. Also run the store tests against a throwaway Postgres 16 (`/usr/lib/postgresql/16/bin`; run `initdb`/`pg_ctl` as a non-root user; port 55432; apply `sql/migrate_app_conversations.sql` then `sql/migrate_app_users.sql`; `TEST_DATABASE_URL=postgresql+psycopg2://…`), then stop and remove it. `python3 -m api.export_openapi`; `cd web && npm run gen:types && npm run typecheck` (fix type fallout in `web/src/lib/types.ts` only as far as needed to compile).
- [ ] **Step 5: Commit** — `feat(api): users, user-owned conversations and preferences`

---

### Task 2: Web — Tailwind, shadcn/ui, theme tokens, light/dark

**Files:** Tailwind v4 + PostCSS config, `web/components.json`, `web/src/lib/utils.ts`, `web/src/app/globals.css` (rewritten), `web/src/components/ui/{button,input,textarea,dropdown-menu,dialog,sheet,tooltip,separator,scroll-area,skeleton,avatar,badge,sonner}.tsx`, `web/src/components/theme-provider.tsx`, `web/src/app/layout.tsx`, `web/tests/unit/theme.test.tsx`. Remove `web/src/app/intel/`, `web/src/components/TopBar.*`, `web/tests/unit/topbar.test.tsx`, `web/tests/e2e/intel.spec.ts`.

**Interfaces — Produces:**
- Design tokens as CSS variables for both modes (shadcn naming: `--background`, `--foreground`, `--card`, `--muted`, `--muted-foreground`, `--border`, `--primary` = Rooster red `#e5484d`, `--primary-foreground`, `--ring`, plus `--up` `--down` `--warn`, and `--font-sans`/`--font-mono`). Light values from the `.L` block and dark from the `.B` block of `docs/design/v4-mockups/ticker.html`.
- The legacy variables the kept components read (`--bg`, `--card`, `--border`, `--text`, `--muted`, `--accent`, transit/listing/VUT colors) are redefined from the new tokens in both modes, so `components/primitives`, `components/map`, `components/intel` render correctly in light and dark without edits. `chartTheme.ts` reads colors from CSS variables at render time (no hardcoded dark hexes).
- `ThemeProvider` (next-themes: `attribute="class"`, `defaultTheme="system"`, `enableSystem`) wrapping the app in `layout.tsx`; `<html lang="es" suppressHydrationWarning>`.
- A `.num` utility class: mono font + `font-variant-numeric: tabular-nums`.
- `/` temporarily renders a minimal page proving tokens and a Button (replaced in Task 4).

- [ ] **Step 1: Failing tests** `theme.test.tsx`: `defaults_to_system_theme` · `toggle_sets_dark_class_on_html` · `legacy_vars_defined_in_both_modes` (parse `globals.css`: every legacy variable appears under `:root` and `.dark`) · `primary_is_rooster_red` . Existing primitive/map/intel unit tests must still pass.
- [ ] **Step 2:** `cd web && npm test` → FAIL. **Step 3: Implement.** 
- [ ] **Step 4:** `npm test && npm run typecheck && npm run lint && npm run build` → PASS (e2e is repaired in Task 7).
- [ ] **Step 5: Commit** — `feat(web): Tailwind + shadcn/ui foundation with light and dark themes`

---

### Task 3: Web — Google sign-in, allowlist, identity through the proxy

**Files:** `web/src/auth.ts`, `web/src/lib/allowlist.ts`, `web/src/app/api/auth/[...nextauth]/route.ts`, the Next 16 request-interception file (`proxy.ts` or `middleware.ts` per the bundled docs), `web/src/app/entrar/page.tsx`, `web/src/app/entrar/invitacion/page.tsx`, `web/src/app/api/rooster/[...path]/route.ts`, `web/.env.example`, tests `web/tests/unit/{allowlist,auth,proxy,entrar}.test.*`.

**Interfaces — Produces:**
- `isAllowed(email: string | null | undefined, raw = process.env.ROOSTER_ALLOWED_EMAILS): boolean`.
- Auth.js config: Google provider (`AUTH_GOOGLE_ID`, `AUTH_GOOGLE_SECRET`, `AUTH_SECRET`), JWT session strategy, `pages.signIn = "/entrar"`. `signIn` callback: not allowed ⇒ redirect `/entrar/invitacion`. On first JWT creation the server calls `PUT ${ROOSTER_API_URL}/users` (with the key) and stores the returned `id` in the token as `uid`; if that call fails, sign-in fails with redirect `/entrar?error=servicio`. `session.user.id = token.uid`.
- Exports from `auth.ts`: `auth`, `signIn`, `signOut`, `handlers`. Helper `currentUserId(): Promise<string | null>`.
- Route protection: unauthenticated requests to anything except `/entrar*`, `/compartir/*`, `/api/auth/*`, Next static assets ⇒ redirect to `/entrar?next=<path>` (pages) — `/api/rooster/*` is handled by the proxy route itself.
- Proxy route: no session ⇒ 401 `{"detail":"signed_out"}` without calling upstream; otherwise sets `x-rooster-user` from the session (never from the incoming request); allowed prefixes become `conversations`, `intel`, `me`; methods GET, POST, PATCH, DELETE; everything else as today.
- `/entrar`: wordmark, one line of value proposition, "Continuar con Google" button (server action `signIn("google", { redirectTo: next ?? "/" })`), error line for `?error=servicio`. `/entrar/invitacion`: "Rooster es solo por invitación" + how to request access + "Probar con otra cuenta".

- [ ] **Step 1: Failing tests:** allowlist — `matches_case_insensitively_and_trims` (`" Ana@Fund.com , b@x.es"` allows `ana@fund.com`) · `empty_or_unset_rejects_everyone` · `null_email_rejected`. auth callbacks (unit, upstream fetch mocked) — `disallowed_email_redirects_to_invitacion` · `allowed_email_upserts_user_and_stores_uid` · `upsert_failure_blocks_sign_in`. proxy — existing tests adapted plus `no_session_401_signed_out_without_upstream_call` · `forwards_user_id_from_session_not_from_request` · `me_prefix_and_patch_delete_forwarded` . entrar — `shows_google_button_and_service_error`.
- [ ] **Step 2:** FAIL. **Step 3: Implement.** **Step 4:** `npm test && npm run typecheck && npm run lint && npm run build` → PASS.
- [ ] **Step 5: Commit** — `feat(web): invite-only Google sign-in and per-user API identity`

---

### Task 4: Web — app shell

**Files:** `web/src/app/(app)/layout.tsx`, `web/src/app/(app)/page.tsx`, `web/src/components/shell/{AppShell,LeftNav,TopBarV4,UserMenu,MobileNav}.tsx`, `web/src/lib/use-preferences.ts`, `web/src/lib/client.ts`, `web/src/lib/copy.ts`, tests `web/tests/unit/{shell,preferences}.test.tsx`.

**Interfaces — Consumes:** `auth()` (Task 3), `/api/rooster/me`, `PATCH /api/rooster/me/preferences` (Task 1), tokens/ui (Task 2). **Produces:**
- `client.ts`: `getMe(): Promise<UserOut>`, `patchPreferences(p: Partial<Preferences>): Promise<Preferences>`; every client call treats 401 as signed out by `window.location.assign("/entrar")`.
- `PreferencesProvider` + `usePreferences(): { prefs: Preferences; set(p: Partial<Preferences>): void }` — initial value from the server (fetched in the `(app)` layout and passed down), optimistic update, PATCH in the background; `theme` changes also call next-themes `setTheme`; on load the stored `theme` (if any) is applied.
- `AppShell({ user, children, chat })`: CSS grid — left nav 220 px, main, optional right slot for the chat (Task 6 fills it; until then the slot is empty). ≥1024 px: nav fixed; below: nav becomes a `Sheet` opened from a menu button in the top bar.
- `LeftNav`: wordmark; items in this order with lucide icons — Inicio `/`, Mapa, Barrios, Anuncios, Comparar, Calculadora, Seguimiento; items whose page doesn't exist yet are rendered disabled (`aria-disabled`, not focusable links) with a "Pronto" badge; then a "Conversaciones" section slot (Task 5 fills it); "Ajustes" is omitted in Phase 1.
- `TopBarV4`: search input rendered disabled with placeholder "Buscar barrio, dirección o anuncio…" (enabled in Phase 2); chat toggle button (wired in Task 6); `UserMenu` (avatar/initials → name, email, theme: Claro / Oscuro / Sistema, "Cerrar sesión").
- Inicio (phase-1 version): greeting with the user's first name, 4 example-question pills, and a "Conversaciones recientes" list slot (Task 5); pills call a `startChat(text)` function provided by Task 5 (until then they link to `/chat?q=<text>`).

- [ ] **Step 1: Failing tests:** `nav_lists_items_in_order_with_future_ones_disabled` · `active_item_marked_aria_current` · `user_menu_shows_name_email_and_signs_out` · `theme_choice_patches_preferences_and_sets_theme` · `preferences_optimistic_then_rollback_on_failure` · `mobile_nav_opens_in_sheet` · `client_401_redirects_to_entrar`.
- [ ] **Step 2:** FAIL. **Step 3: Implement.** **Step 4:** `npm test && npm run typecheck && npm run lint && npm run build` → PASS.
- [ ] **Step 5: Commit** — `feat(web): signed-in app shell with navigation, user menu and theme preference`

---

### Task 5: Web — conversations data layer and history

**Files:** `web/src/lib/{client.ts,use-chat.ts,chat-state.ts,chat-context.tsx,use-conversations.ts,types.ts,copy.ts}`, `web/src/components/chat/ConversationList.tsx`, `web/src/app/(app)/chat/page.tsx`, `LeftNav` + Inicio slots, tests `web/tests/unit/{conversations,use-chat,chat-state,client}.test.*`.

**Interfaces — Produces:**
- `client.ts`: `listConversations(): Promise<ConversationSummary[]>`, `renameConversation(id, title)`, `deleteConversation(id)`.
- `ConversationsProvider` + `useConversations(): { items: ConversationSummary[]; loading: boolean; refresh(): Promise<void>; rename(id, title): Promise<void>; remove(id): Promise<void> }` — optimistic rename/remove with rollback; `refresh` is called by the chat hook after a turn finishes and after creating a conversation.
- Chat hook changes: the active conversation id is held in `ChatProvider` (mounted in the `(app)` layout, inside `ConversationsProvider`) and mirrored to `localStorage` under `rooster.conversationId.<userId>`; `open(id)` loads another conversation (aborting nothing if a turn is streaming in the current one is not allowed: `open` and `reset` are ignored while streaming and the UI disables them); `startChat(text)` = reset + send; a 404 on load clears the stored id; removing the active conversation resets the chat.
- `ConversationList({ variant: "nav" | "page" })`: nav shows the 8 most recent titles (untitled ⇒ "Nueva conversación"), active one marked; page (`/chat`) shows all with relative date, rename (inline input, Enter saves, Esc cancels) and delete (confirm dialog). Clicking an item calls `open(id)`; on the page it also navigates to `/chat/<id>`.

- [ ] **Step 1: Failing tests:** `lists_recent_in_nav_max_8` · `untitled_shows_placeholder` · `open_loads_and_marks_active` · `open_and_reset_ignored_while_streaming` · `rename_optimistic_and_rollback` · `delete_active_resets_chat` · `stale_id_404_cleared` (per-user key) · `storage_key_is_per_user` · `refresh_after_turn_finishes` · `start_chat_resets_then_sends`.
- [ ] **Step 2:** FAIL. **Step 3: Implement.** **Step 4:** `npm test && npm run typecheck && npm run lint && npm run build` → PASS.
- [ ] **Step 5: Commit** — `feat(web): per-user conversation history with rename and delete`

---

### Task 6: Web — chat surfaces (sidebar, full screen, phone sheet)

**Files:** `web/src/components/chat/{ChatPanel,ChatSidebar,ChatSheet,ChatHeader,MessageList,AssistantMessage,Composer,ThinkingBox,FollowUps,FallbackCard,CompactBlock,BlockDialog}.tsx` (rebuilt on Tailwind/shadcn; old `*.module.css` and `ChatView` removed), `web/src/app/(app)/chat/[id]/page.tsx`, `AppShell`/`TopBarV4` wiring, tests `web/tests/unit/chat*.test.tsx`.

**Interfaces — Consumes:** `useChatContext`, `useConversations`, `usePreferences`, `Primitive`. **Produces:**
- `ChatPanel({ density: "compact" | "full" })` — the one chat UI (header, message list, composer) used by all three surfaces; all v3 behaviors kept (streaming text, thinking box, follow-up pills only on the latest answer, busy and rate-limit notices with restored draft, failure card, auto-follow, 2,000-char limit, disabled while loading/streaming, `data-testid="message-block"`).
- `ChatSidebar` — right column 380 px on ≥1024 px, shown when `prefs.chat_open !== false` and the route is not `/chat/*`; header has title, "Nueva conversación", expand (→ `/chat/<id>` or `/chat` when none) and close (sets `chat_open: false`). The top-bar chat button toggles it.
- `ChatSheet` — below 1024 px the chat opens from a floating button as a bottom `Sheet` (85 % height) with an expand action to the full-screen route.
- `/chat/[id]` — full-width `ChatPanel density="full"`; opens that conversation (404 ⇒ redirect `/chat`); a "Volver" action returns to the previous page with the sidebar open.
- Compact density: `text`, `kpi` and tables with ≤ 4 columns and ≤ 6 rows render inline; every other block renders as `CompactBlock` (a card with the block's title/kind label and a small non-interactive preview for charts; a placeholder icon for maps/code) that opens `BlockDialog` (large dialog rendering the full `Primitive`). Full density renders every block inline as in v3.

- [ ] **Step 1: Failing tests:** `sidebar_visible_by_default_and_closes_to_preference` · `sidebar_hidden_on_full_screen_route` · `expand_navigates_to_chat_id` · `same_live_turn_shown_after_switching_surface` (Review Focus 5: provider state with a streaming turn renders in sidebar, then in full page, without a new request) · `compact_inlines_small_table_and_cards_big_blocks` · `compact_card_opens_full_block_dialog` · `sheet_used_below_1024` · `full_route_404_redirects_to_chat` · the v3 chat behavior tests ported to `ChatPanel`.
- [ ] **Step 2:** FAIL. **Step 3: Implement.** **Step 4:** `npm test && npm run typecheck && npm run lint && npm run build` → PASS.
- [ ] **Step 5: Commit** — `feat(web): chat as sidebar, full screen and phone sheet with compact answers`

---

### Task 7: End-to-end, local stack, docs

**Files:** `web/tests/fake-api/server.mjs`, `web/tests/e2e/{helpers.ts,auth.spec.ts,chat.spec.ts,history.spec.ts,mobile.spec.ts}`, `web/playwright.config.ts`, `docker-compose.yml`, `.env.example`, `web/.env.example`, `README.md`, `.claude/CLAUDE.md`, `docs/concept.md` (accounts paragraph only).

**Interfaces:**
- Fake API: implements Task 1's routes with per-user conversation storage keyed by `x-rooster-user`, `PUT /users`, `GET /me`, `PATCH /me/preferences`; turns as in v3.
- Signed-in session for e2e **without any test mode in production code**: a Playwright helper mints a valid Auth.js session cookie with `encode` from `next-auth/jwt` using the test `AUTH_SECRET` (token carries `uid`, `email`, `name`) and adds it to the browser context. Web server env for e2e: `AUTH_SECRET=test-secret`, `AUTH_URL=http://127.0.0.1:3100`, dummy Google id/secret, `ROOSTER_ALLOWED_EMAILS=ana@fund.com`.
- Compose/env: web service gains `AUTH_SECRET`, `AUTH_URL`, `AUTH_GOOGLE_ID`, `AUTH_GOOGLE_SECRET`, `ROOSTER_ALLOWED_EMAILS` (required, documented); README: creating the Google OAuth client (redirect URI `<AUTH_URL>/api/auth/callback/google`), the allowlist, the new migration, and that existing v3 conversations are deleted by it.

- [ ] **Step 1: Write e2e specs:** auth — `signed_out_redirects_to_entrar` · `entrar_shows_google_button` · `invitacion_page_renders`. chat — `sidebar_turn_renders_compact_then_full_dialog` · `expand_to_full_screen_keeps_conversation` · `leaving_mid_turn_keeps_the_answer` (kept from v3) · `error_turn_shows_fallback` · `theme_toggle_persists_after_reload`. history — `new_conversation_appears_in_nav_with_title` · `rename_and_delete_from_chat_page` · `second_user_cannot_see_first_users_conversation` (two cookies). mobile (390×844) — `no_horizontal_scroll_on_inicio_and_chat` · `chat_opens_as_sheet_and_sends`.
- [ ] **Step 2:** `PLAYWRIGHT_CHROMIUM_EXECUTABLE=/opt/pw-browsers/chromium-1243/chrome-linux64/chrome npm run e2e` → iterate until PASS.
- [ ] **Step 3:** compose + env + docs; `docker compose config` validates.
- [ ] **Step 4: Run everything:** `python3 -m pytest -q`; `cd web && npm test && npm run typecheck && npm run lint && npm run check:types && npm run build && npm run e2e` → all PASS.
- [ ] **Step 5: Commit** — `test(web): v4 foundation end-to-end tests` then `chore: v4 foundation env, compose and docs`
