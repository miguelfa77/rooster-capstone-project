-- Conversation storage for the Rooster API (v3). Idempotent.
CREATE SCHEMA IF NOT EXISTS app;

CREATE TABLE IF NOT EXISTS app.conversations (
    id                          uuid PRIMARY KEY,
    created_at                  timestamptz NOT NULL DEFAULT now(),
    updated_at                  timestamptz NOT NULL DEFAULT now(),
    prompt_cache_key            text NOT NULL,
    session_memory              jsonb NOT NULL DEFAULT '{}'::jsonb,
    previous_execution_results  jsonb NOT NULL DEFAULT '[]'::jsonb
);

CREATE TABLE IF NOT EXISTS app.messages (
    id               bigserial PRIMARY KEY,
    conversation_id  uuid NOT NULL REFERENCES app.conversations(id) ON DELETE CASCADE,
    position         int NOT NULL,
    body             jsonb NOT NULL,
    created_at       timestamptz NOT NULL DEFAULT now(),
    UNIQUE (conversation_id, position)
);
