-- =============================================================================
-- feat_transaction_signals.sql
-- Plan, payment, and subscription lifecycle features from transactions.
--
-- Output features per user:
--   auto_renew_flag            - current auto-renew setting (last transaction)
--   is_cancel_flag             - has cancellation on record
--   plan_downgrade_flag        - ever downgraded plan
--   payment_failure_count_90d  - payment failures in last 90 days
--   days_since_last_txn        - recency of last transaction
--   membership_expire_days     - days until subscription expires
--   avg_discount_rate          - how much discount they've historically received
--   payment_method_changes     - number of distinct payment methods used
--   total_paid_90d             - total revenue in last 90 days
-- =============================================================================

CREATE OR REPLACE TABLE feat_transaction_signals AS

WITH ref AS (
    SELECT DATE '2017-03-31' AS ref_date
),

enriched AS (
    SELECT
        t.*,
        DATE_DIFF('day', t.transaction_date, r.ref_date) AS days_ago

    FROM stg_transactions t
    CROSS JOIN ref r
),

-- Most recent transaction per user (for point-in-time features)
latest_txn AS (
    SELECT DISTINCT ON (msno)
        msno,
        transaction_date         AS last_txn_date,
        membership_expire_date,
        is_auto_renew            AS auto_renew_flag,
        is_cancel                AS is_cancel_flag,
        plan_list_price          AS current_plan_price

    FROM enriched
    ORDER BY msno, transaction_date DESC
)

SELECT
    e.msno,

    -- Point-in-time from latest transaction
    l.auto_renew_flag,
    l.is_cancel_flag,
    DATE_DIFF('day', l.last_txn_date,         r.ref_date) AS days_since_last_txn,
    DATE_DIFF('day', r.ref_date, l.membership_expire_date) AS membership_expire_days,

    -- Plan downgrade: ever had a downgrade in transaction history
    MAX(e.plan_downgrade_flag)                             AS plan_downgrade_flag,

    -- Payment failures in last 90 days
    COUNT(CASE
        WHEN e.payment_failure_flag = 1 AND e.days_ago <= 90
        THEN 1 END)                                        AS payment_failure_count_90d,

    -- Discount behavior
    ROUND(AVG(e.discount_rate), 4)                         AS avg_discount_rate,

    MAX(CASE
        WHEN e.discount_rate > 0 AND e.days_ago <= 90
        THEN 1 ELSE 0
    END)                                                   AS received_discount_90d,

    -- Payment method diversity (switching = instability signal)
    COUNT(DISTINCT e.payment_method_id)                    AS payment_method_changes,

    -- Revenue
    SUM(CASE
        WHEN e.days_ago <= 90
        THEN e.actual_amount_paid ELSE 0
    END)                                                   AS total_paid_90d,

    -- Plan stability: how many distinct plan prices they've been on
    COUNT(DISTINCT e.plan_list_price)                      AS distinct_plan_count,

    -- Cancellation count
    COUNT(CASE WHEN e.is_cancel = 1 THEN 1 END)            AS cancellation_count

FROM enriched e
LEFT JOIN latest_txn l USING (msno)
CROSS JOIN ref r
GROUP BY
    e.msno,
    l.auto_renew_flag,
    l.is_cancel_flag,
    l.last_txn_date,
    l.membership_expire_date,
    r.ref_date;
