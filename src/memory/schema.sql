-- Delta tables backing agent memory. Run once against the target
-- catalog/schema in config.yaml -> memory before first app launch.

CREATE CATALOG IF NOT EXISTS ai_projects_catalog;
CREATE SCHEMA IF NOT EXISTS ai_projects_catalog.agent_demo;

-- Short-term: rolling per-session turn history used to build conversational
-- context. Trimmed by the app to `short_term_turns_limit` on read.
CREATE TABLE IF NOT EXISTS ai_projects_catalog.agent_demo.clinical_chat_short_term_memory (
    session_id      STRING      NOT NULL,
    turn_id         BIGINT      NOT NULL,
    role            STRING      NOT NULL,   -- 'user' | 'assistant'
    content         STRING      NOT NULL,
    sql_query       STRING,                 -- populated on assistant turns that ran SQL
    created_at      TIMESTAMP   NOT NULL
) USING DELTA;

-- Long-term: durable user preferences extracted across sessions
-- (e.g. "prefers backpacks under 2000 INR"). One row per user/preference.
CREATE TABLE IF NOT EXISTS ai_projects_catalog.agent_demo.clinical_chat_long_term_memory (
    user_id         STRING      NOT NULL,
    preference_key  STRING      NOT NULL,
    preference_value STRING     NOT NULL,
    updated_at      TIMESTAMP   NOT NULL
) USING DELTA;
