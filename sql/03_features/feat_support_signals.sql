-- =============================================================================
-- feat_support_signals.sql
-- Support ticket velocity, sentiment, and category features.
--
-- These are the strongest behavioral churn signals alongside usage decay.
-- Churners spike in ticket volume, negative sentiment, and cancellation
-- inquiries in the 14-30 days before they leave.
--
-- Output features per user:
--   ticket_count_30d          - tickets submitted in last 30 days
--   ticket_count_90d          - tickets submitted in last 90 days
--   ticket_velocity           - ratio of 30d to 90d ticket rate (spike detector)
--   pct_negative_sentiment    - % of tickets with negative sentiment (all time)
--   unresolved_ticket_count   - open unresolved tickets
--   has_cancellation_inquiry  - ever submitted a cancellation inquiry (binary)
--   cancellation_inquiry_30d  - cancellation inquiry in last 30 days (stronger)
--   avg_time_to_fix_hours     - avg resolution time (service quality signal)
--   billing_ticket_count      - billing-related tickets (payment frustration)
-- =============================================================================

CREATE OR REPLACE TABLE feat_support_signals AS

WITH ref AS (
    SELECT DATE '2017-03-31' AS ref_date
),

enriched AS (
    SELECT
        t.*,
        DATE_DIFF('day', t.ticket_date, r.ref_date) AS days_ago

    FROM stg_support_tickets t
    CROSS JOIN ref r
)

SELECT
    msno,

    -- Volume: rolling windows
    COUNT(CASE WHEN days_ago <= 30  THEN 1 END)     AS ticket_count_30d,
    COUNT(CASE WHEN days_ago <= 90  THEN 1 END)     AS ticket_count_90d,
    COUNT(*)                                         AS ticket_count_all,

    -- Velocity: spike in recent period vs baseline
    -- > 1 means tickets are accelerating (churn signal)
    ROUND(
        COUNT(CASE WHEN days_ago <= 30 THEN 1 END) * 3.0
        / NULLIF(COUNT(CASE WHEN days_ago <= 90 THEN 1 END), 0)
    , 4) AS ticket_velocity,

    -- Sentiment
    ROUND(
        AVG(is_negative::FLOAT)
    , 4) AS pct_negative_sentiment,

    COUNT(CASE WHEN is_negative = 1 THEN 1 END)     AS negative_ticket_count,

    -- Resolution
    COUNT(CASE WHEN is_unresolved = 1 THEN 1 END)   AS unresolved_ticket_count,

    ROUND(
        AVG(CASE WHEN time_to_fix_hours IS NOT NULL
            THEN time_to_fix_hours END)
    , 2) AS avg_time_to_fix_hours,

    -- Cancellation signals
    MAX(is_cancellation)                             AS has_cancellation_inquiry,

    MAX(CASE
        WHEN is_cancellation = 1 AND days_ago <= 30
        THEN 1 ELSE 0
    END)                                             AS cancellation_inquiry_30d,

    -- Billing frustration
    COUNT(CASE WHEN category = 'billing' THEN 1 END) AS billing_ticket_count,

    COUNT(CASE
        WHEN category = 'billing' AND days_ago <= 30
        THEN 1 END)                                  AS billing_ticket_30d

FROM enriched
GROUP BY msno;
