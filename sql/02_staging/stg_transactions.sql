-- =============================================================================
-- stg_transactions.sql
-- Clean and type-cast the raw transactions table.
--
-- Key transforms:
--   - Parse all date columns (integer YYYYMMDD → DATE)
--   - Derive subscription duration in days
--   - Flag payment failures (amount_paid = 0 but not a free plan)
--   - Flag plan cancellations and auto-renew status
--   - Detect plan downgrades (plan_list_price drops across consecutive rows)
-- =============================================================================

CREATE OR REPLACE TABLE stg_transactions AS

WITH parsed AS (
    SELECT
        msno,
        payment_method_id,
        payment_plan_days,
        plan_list_price,
        actual_amount_paid,
        is_auto_renew,
        is_cancel,

        TRY_CAST(
            STRPTIME(CAST(transaction_date    AS VARCHAR), '%Y%m%d')
        AS DATE) AS transaction_date,

        TRY_CAST(
            STRPTIME(CAST(membership_expire_date AS VARCHAR), '%Y%m%d')
        AS DATE) AS membership_expire_date

    FROM raw_transactions
    WHERE msno IS NOT NULL
),

with_derived AS (
    SELECT
        *,

        -- Days until expiry from transaction date
        DATE_DIFF('day', transaction_date, membership_expire_date)
            AS days_until_expiry,

        -- Payment failure: paid nothing but plan has a price
        CASE
            WHEN actual_amount_paid = 0 AND plan_list_price > 0 THEN 1
            ELSE 0
        END AS payment_failure_flag,

        -- Discount rate: how much off list price did they pay
        CASE
            WHEN plan_list_price > 0
                THEN ROUND(1.0 - actual_amount_paid / plan_list_price, 4)
            ELSE 0
        END AS discount_rate,

        -- Lag plan price to detect downgrades (lower price = simpler plan)
        LAG(plan_list_price) OVER (
            PARTITION BY msno
            ORDER BY transaction_date
        ) AS prev_plan_list_price

    FROM parsed
),

with_downgrade AS (
    SELECT
        *,
        CASE
            WHEN prev_plan_list_price IS NOT NULL
             AND plan_list_price < prev_plan_list_price THEN 1
            ELSE 0
        END AS plan_downgrade_flag

    FROM with_derived
)

SELECT * FROM with_downgrade;
