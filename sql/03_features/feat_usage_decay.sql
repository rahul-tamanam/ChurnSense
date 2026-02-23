-- =============================================================================
-- feat_usage_decay.sql
-- Rolling usage metrics and decay slopes across 30 / 60 / 90 day windows.
--
-- "Decay" = the slope of usage over time within the window (negative = declining).
-- We approximate slope using (recent_avg - earlier_avg) / earlier_avg.
--
-- Reference date = 2017-03-31 (KKBox prediction window cutoff).
--
-- Output features per user:
--   sessions_last_Xd          - number of active days in window
--   hours_last_Xd             - total listening hours in window
--   tracks_last_Xd            - total tracks played in window
--   avg_daily_hours_last_Xd   - mean daily listening hours
--   usage_decay_Xd            - % change: first half vs second half of window
--   completion_rate_avg_Xd    - avg track completion rate
--   skip_rate_avg_Xd          - avg track skip rate
-- =============================================================================

CREATE OR REPLACE TABLE feat_usage_decay AS

WITH ref AS (
    SELECT DATE '2017-03-31' AS ref_date
),

-- Pre-filter to only the rows we need (last 90 days)
windowed AS (
    SELECT
        l.msno,
        l.log_date,
        l.listening_hours,
        l.total_tracks,
        l.completion_rate,
        l.skip_rate,
        DATE_DIFF('day', l.log_date, r.ref_date) AS days_ago

    FROM stg_user_logs l
    CROSS JOIN ref r
    WHERE l.log_date >= (r.ref_date - INTERVAL '90 days')
),

-- 30-day window
w30 AS (
    SELECT
        msno,
        COUNT(DISTINCT log_date)          AS sessions_last_30d,
        SUM(listening_hours)              AS hours_last_30d,
        SUM(total_tracks)                 AS tracks_last_30d,
        AVG(listening_hours)              AS avg_daily_hours_last_30d,
        AVG(completion_rate)              AS completion_rate_avg_30d,
        AVG(skip_rate)                    AS skip_rate_avg_30d,

        -- Decay: compare first 15 days vs last 15 days of the window
        -- Positive decay_30d means usage DROPPED (churn signal)
        ROUND(
            (AVG(CASE WHEN days_ago BETWEEN 16 AND 30 THEN listening_hours END) -
             AVG(CASE WHEN days_ago BETWEEN  1 AND 15 THEN listening_hours END))
            / NULLIF(AVG(CASE WHEN days_ago BETWEEN 16 AND 30 THEN listening_hours END), 0)
        , 4) AS usage_decay_30d

    FROM windowed
    WHERE days_ago <= 30
    GROUP BY msno
),

-- 60-day window
w60 AS (
    SELECT
        msno,
        COUNT(DISTINCT log_date)          AS sessions_last_60d,
        SUM(listening_hours)              AS hours_last_60d,
        SUM(total_tracks)                 AS tracks_last_60d,
        AVG(listening_hours)              AS avg_daily_hours_last_60d,

        ROUND(
            (AVG(CASE WHEN days_ago BETWEEN 31 AND 60 THEN listening_hours END) -
             AVG(CASE WHEN days_ago BETWEEN  1 AND 30 THEN listening_hours END))
            / NULLIF(AVG(CASE WHEN days_ago BETWEEN 31 AND 60 THEN listening_hours END), 0)
        , 4) AS usage_decay_60d

    FROM windowed
    WHERE days_ago <= 60
    GROUP BY msno
),

-- 90-day window
w90 AS (
    SELECT
        msno,
        COUNT(DISTINCT log_date)          AS sessions_last_90d,
        SUM(listening_hours)              AS hours_last_90d,
        SUM(total_tracks)                 AS tracks_last_90d,
        AVG(listening_hours)              AS avg_daily_hours_last_90d,

        ROUND(
            (AVG(CASE WHEN days_ago BETWEEN 46 AND 90 THEN listening_hours END) -
             AVG(CASE WHEN days_ago BETWEEN  1 AND 45 THEN listening_hours END))
            / NULLIF(AVG(CASE WHEN days_ago BETWEEN 46 AND 90 THEN listening_hours END), 0)
        , 4) AS usage_decay_90d

    FROM windowed
    WHERE days_ago <= 90
    GROUP BY msno
)

-- Join all windows together
SELECT
    COALESCE(w30.msno, w60.msno, w90.msno) AS msno,

    -- 30d
    COALESCE(w30.sessions_last_30d, 0)        AS sessions_last_30d,
    COALESCE(w30.hours_last_30d, 0)            AS hours_last_30d,
    COALESCE(w30.tracks_last_30d, 0)           AS tracks_last_30d,
    COALESCE(w30.avg_daily_hours_last_30d, 0)  AS avg_daily_hours_last_30d,
    COALESCE(w30.completion_rate_avg_30d, 0)   AS completion_rate_avg_30d,
    COALESCE(w30.skip_rate_avg_30d, 0)         AS skip_rate_avg_30d,
    w30.usage_decay_30d,

    -- 60d
    COALESCE(w60.sessions_last_60d, 0)        AS sessions_last_60d,
    COALESCE(w60.hours_last_60d, 0)            AS hours_last_60d,
    COALESCE(w60.tracks_last_60d, 0)           AS tracks_last_60d,
    COALESCE(w60.avg_daily_hours_last_60d, 0)  AS avg_daily_hours_last_60d,
    w60.usage_decay_60d,

    -- 90d
    COALESCE(w90.sessions_last_90d, 0)        AS sessions_last_90d,
    COALESCE(w90.hours_last_90d, 0)            AS hours_last_90d,
    COALESCE(w90.tracks_last_90d, 0)           AS tracks_last_90d,
    COALESCE(w90.avg_daily_hours_last_90d, 0)  AS avg_daily_hours_last_90d,
    w90.usage_decay_90d

FROM w30
FULL OUTER JOIN w60 USING (msno)
FULL OUTER JOIN w90 USING (msno);
