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
    fm.msno,

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

    -- ── All churn features (same as mart_churn_features) ─────────────────────
    fm.usage_decay_30d,
    fm.usage_decay_60d,
    fm.usage_decay_90d,
    fm.sessions_last_30d,
    fm.sessions_last_60d,
    fm.sessions_last_90d,
    fm.avg_session_duration_30d,

    fm.login_days_last_30d,
    fm.login_gap_days_avg,
    fm.login_streak_max,
    fm.days_since_last_login,

    fm.ticket_count_30d,
    fm.ticket_count_90d,
    fm.ticket_velocity,
    fm.pct_negative_sentiment,
    fm.unresolved_ticket_count,
    fm.has_cancellation_inquiry,

    fm.auto_renew_flag,
    fm.plan_downgrade_flag,
    fm.payment_failure_count_90d,
    fm.days_since_last_txn,
    fm.avg_discount_rate,

    fm.unique_features_used_30d,
    fm.feature_diversity_score,
    fm.days_since_last_feature_use,

    fm.change_point_detected,
    fm.days_since_change_point,
    fm.usage_slope_post_change,

    fm.gender_male_flag,
    fm.city_encoded,
    fm.age_bucket_encoded,
    fm.tenure_days,

    fm.behavioral_cohort,
    fm.risk_tier

FROM feat_master fm
INNER JOIN stg_interventions si ON fm.msno = si.msno
INNER JOIN stg_outcomes      so ON fm.msno = so.msno
                                AND si.intervention_type = so.intervention_type;
