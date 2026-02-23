-- =============================================================================
-- mart_churn_features.sql
-- Final ML-ready feature mart with train/score split.
--
-- Train split  = users in raw_train with is_churn label
-- Score split  = all users (for inference — label unknown in production)
--
-- Also applies final transformations:
--   - Clip extreme values (decay rates, gap days)
--   - Encode categorical columns as integers
--   - Add behavioral cohort labels
-- =============================================================================

CREATE OR REPLACE TABLE mart_churn_features AS

WITH clipped AS (
    SELECT
        *,

        -- Clip decay metrics: values outside [-2, 2] are noise
        GREATEST(-2, LEAST(2, COALESCE(usage_decay_30d, 0))) AS usage_decay_30d_clipped,
        GREATEST(-2, LEAST(2, COALESCE(usage_decay_60d, 0))) AS usage_decay_60d_clipped,
        GREATEST(-2, LEAST(2, COALESCE(usage_decay_90d, 0))) AS usage_decay_90d_clipped,

        -- Clip gap days: cap at 180 (half a year absence = same signal as full year)
        LEAST(180, login_gap_days_max)     AS login_gap_days_max_clipped,
        LEAST(180, days_since_last_login)  AS days_since_last_login_clipped,
        LEAST(180, days_since_last_txn)    AS days_since_last_txn_clipped

    FROM feat_master
),

encoded AS (
    SELECT
        *,

        -- Age bucket → integer ordinal
        CASE age_bucket
            WHEN '10-19'  THEN 1
            WHEN '20-29'  THEN 2
            WHEN '30-39'  THEN 3
            WHEN '40-49'  THEN 4
            WHEN '50+'    THEN 5
            ELSE 0
        END AS age_bucket_encoded,

        -- Behavioral cohort: segment users by engagement pattern
        CASE
            WHEN sessions_last_30d >= 20
             AND usage_decay_30d_clipped > -0.2  THEN 'power_user'
            WHEN sessions_last_30d >= 10          THEN 'regular'
            WHEN sessions_last_30d >= 1           THEN 'casual'
            WHEN days_since_last_login <= 30      THEN 'recently_inactive'
            ELSE                                       'dormant'
        END AS behavioral_cohort,

        -- Risk tier: simple rule-based pre-segmentation for interpretability
        CASE
            WHEN usage_decay_30d_clipped > 0.3
             AND ticket_count_30d >= 2            THEN 'critical'
            WHEN usage_decay_30d_clipped > 0.2
             OR  has_cancellation_inquiry = 1     THEN 'high'
            WHEN sessions_last_30d < 5
             AND days_since_last_login > 14       THEN 'medium'
            ELSE                                       'low'
        END AS risk_tier

    FROM clipped
)

SELECT
    msno,
    is_churn,

    -- ── Final clipped features ────────────────────────────────────────────────
    usage_decay_30d_clipped         AS usage_decay_30d,
    usage_decay_60d_clipped         AS usage_decay_60d,
    usage_decay_90d_clipped         AS usage_decay_90d,

    sessions_last_30d,
    sessions_last_60d,
    sessions_last_90d,
    avg_session_duration_30d,
    completion_rate_avg_30d,
    skip_rate_avg_30d,

    login_days_last_30d,
    login_gap_days_avg,
    login_gap_days_max_clipped      AS login_gap_days_max,
    login_streak_max,
    login_streak_current,
    days_since_last_login_clipped   AS days_since_last_login,

    ticket_count_30d,
    ticket_count_90d,
    ticket_velocity,
    pct_negative_sentiment,
    unresolved_ticket_count,
    has_cancellation_inquiry,
    cancellation_inquiry_30d,

    auto_renew_flag,
    plan_downgrade_flag,
    payment_failure_count_90d,
    days_since_last_txn_clipped     AS days_since_last_txn,
    membership_expire_days,
    avg_discount_rate,
    cancellation_count,

    unique_features_used_30d,
    feature_diversity_score,
    days_since_last_feature_use,

    change_point_detected,
    days_since_change_point,
    usage_slope_post_change,
    pct_usage_change,

    -- Encoded categoricals
    gender_male_flag,
    city                            AS city_encoded,
    age_bucket_encoded,
    tenure_days,

    -- Cohorts and tiers (for analysis, not fed into model directly)
    behavioral_cohort,
    risk_tier,

    -- Train/score split flag
    'train'                         AS split

FROM encoded;
