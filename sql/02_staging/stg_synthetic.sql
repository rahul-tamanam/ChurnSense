-- =============================================================================
-- stg_synthetic.sql
-- Clean and type-cast all four synthetic tables.
--
-- Tables produced:
--   stg_feature_events    - SaaS feature usage events
--   stg_support_tickets   - Support interactions
--   stg_interventions     - A/B treatment assignments
--   stg_outcomes          - Intervention outcomes (for uplift model)
-- =============================================================================


-- ── Feature Events ────────────────────────────────────────────────────────────
CREATE OR REPLACE TABLE stg_feature_events AS

SELECT
    msno,
    TRY_CAST(event_date AS DATE)          AS event_date,
    LOWER(TRIM(feature_name))             AS feature_name,
    COALESCE(session_duration_sec, 0)     AS session_duration_sec

FROM raw_feature_events
WHERE msno IS NOT NULL
  AND event_date IS NOT NULL;


-- ── Support Tickets ───────────────────────────────────────────────────────────
CREATE OR REPLACE TABLE stg_support_tickets AS

SELECT
    ticket_id,
    msno,
    TRY_CAST(ticket_date AS DATE)         AS ticket_date,
    LOWER(TRIM(category))                 AS category,
    LOWER(TRIM(sentiment))                AS sentiment,
    LOWER(TRIM(resolution_status))        AS resolution_status,
    time_to_fix_hours,

    -- Convenience flags used in feature engineering
    CASE WHEN sentiment = 'negative'              THEN 1 ELSE 0 END AS is_negative,
    CASE WHEN resolution_status = 'unresolved'    THEN 1 ELSE 0 END AS is_unresolved,
    CASE WHEN category = 'cancellation_inquiry'   THEN 1 ELSE 0 END AS is_cancellation

FROM raw_support_tickets
WHERE msno IS NOT NULL
  AND ticket_date IS NOT NULL;


-- ── Interventions ─────────────────────────────────────────────────────────────
CREATE OR REPLACE TABLE stg_interventions AS

SELECT
    msno,
    LOWER(TRIM(intervention_type))        AS intervention_type,
    TRY_CAST(intervention_date AS DATE)   AS intervention_date,
    COALESCE(email_opened,  false)        AS email_opened,
    COALESCE(email_clicked, false)        AS email_clicked

FROM raw_interventions
WHERE msno IS NOT NULL;


-- ── Outcomes ──────────────────────────────────────────────────────────────────
CREATE OR REPLACE TABLE stg_outcomes AS

SELECT
    msno,
    LOWER(TRIM(intervention_type))        AS intervention_type,
    COALESCE(renewed,        false)       AS renewed,
    COALESCE(was_saved,      false)       AS was_saved,
    COALESCE(observed_churn, false)       AS observed_churn

FROM raw_outcomes
WHERE msno IS NOT NULL;
