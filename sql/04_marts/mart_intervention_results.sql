-- =============================================================================
-- mart_intervention_results.sql
-- Training data for the uplift (T-learner) model.
--
-- Joins:
--   - feat_master (all user features)
--   - stg_interventions (what treatment they received)
--   - stg_outcomes (did they churn after treatment)
--
-- One row per user. The uplift model trains two XGBoost models on this:
--   model_treatment: trained on rows where intervention_type != 'no_treatment'
--   model_control:   trained on rows where intervention_type = 'no_treatment'
--
-- Output:
--   All features from feat_master
--   + intervention_type
--   + intervention_type_encoded (integer for model input)
--   + email_opened, email_clicked
--   + observed_churn (the outcome label)
--   + was_saved (churner who converted after treatment)
-- =============================================================================

CREATE OR REPLACE TABLE mart_intervention_results AS

SELECT
    -- Identity
    mc.msno,

    -- Intervention assignment
    si.intervention_type,

    -- Encode intervention type as integer for model input
    CASE si.intervention_type
        WHEN 'discount_offer'          THEN 1
        WHEN 'feature_highlight_email' THEN 2
        WHEN 'personal_outreach'       THEN 3
        WHEN 're_onboarding_flow'      THEN 4
        WHEN 'no_treatment'            THEN 0
        ELSE 0
    END AS intervention_type_encoded,

    -- Email engagement signals
    si.email_opened,
    si.email_clicked,

    -- Outcome labels
    so.observed_churn,
    so.was_saved,
    so.renewed,

    -- ── All churn features (clipped/encoded — from mart_churn_features) ────────
    mc.usage_decay_30d,
    mc.usage_decay_60d,
    mc.usage_decay_90d,
    mc.sessions_last_30d,
    mc.sessions_last_60d,
    mc.sessions_last_90d,
    mc.avg_session_duration_30d,

    mc.login_days_last_30d,
    mc.login_gap_days_avg,
    mc.login_streak_max,
    mc.days_since_last_login,

    mc.ticket_count_30d,
    mc.ticket_count_90d,
    mc.ticket_velocity,
    mc.pct_negative_sentiment,
    mc.unresolved_ticket_count,
    mc.has_cancellation_inquiry,

    mc.auto_renew_flag,
    mc.plan_downgrade_flag,
    mc.payment_failure_count_90d,
    mc.days_since_last_txn,
    mc.avg_discount_rate,

    mc.unique_features_used_30d,
    mc.feature_diversity_score,
    mc.days_since_last_feature_use,

    mc.change_point_detected,
    mc.days_since_change_point,
    mc.usage_slope_post_change,

    mc.gender_male_flag,
    mc.city_encoded,
    mc.age_bucket_encoded,
    mc.tenure_days,

    -- Cohort / tier labels (analysis only — not fed into uplift model directly)
    mc.behavioral_cohort,
    mc.risk_tier

FROM mart_churn_features mc
INNER JOIN stg_interventions si ON mc.msno = si.msno
INNER JOIN stg_outcomes      so ON mc.msno = so.msno
                                AND si.intervention_type = so.intervention_type;

