-- Users and user-owned conversations for the Rooster API (v4). Idempotent.
-- Apply after migrate_app_conversations.sql. Deletes conversations that have no owner.

CREATE TABLE IF NOT EXISTS app.users (
    id            uuid PRIMARY KEY,
    email         text NOT NULL UNIQUE,  -- stored lowercased
    name          text,
    image         text,
    preferences   jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at    timestamptz NOT NULL DEFAULT now(),
    last_seen_at  timestamptz NOT NULL DEFAULT now()
);

ALTER TABLE app.conversations
    ADD COLUMN IF NOT EXISTS user_id uuid REFERENCES app.users(id) ON DELETE CASCADE;
ALTER TABLE app.conversations ADD COLUMN IF NOT EXISTS title text;

DELETE FROM app.conversations WHERE user_id IS NULL;
ALTER TABLE app.conversations ALTER COLUMN user_id SET NOT NULL;

CREATE INDEX IF NOT EXISTS conversations_user_updated_idx
    ON app.conversations (user_id, updated_at DESC);
