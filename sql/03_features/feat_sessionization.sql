-- =============================================================================
-- feat_sessionization.sql
-- Group consecutive active days into "sessions" per user.
--
-- A session = a streak of consecutive active days.
-- A new session starts when there's a gap of > 1 day between log entries.
--
-- Output: one row per user-session with:
--   - session start/end dates
--   - session length in days
--   - total listening hours in session
--   - average daily tracks in session
--
-- This is used by feat_login_frequency.sql to compute gap and streak features.
-- =============================================================================

CREATE OR REPLACE TABLE feat_sessions AS

WITH daily_activity AS (
    -- One row per user per active day
    SELECT
        msno,
        log_date,
        SUM(listening_hours) AS daily_hours,
        SUM(total_tracks)    AS daily_tracks

    FROM stg_user_logs
    GROUP BY msno, log_date
),

with_prev_date AS (
    SELECT
        *,
        LAG(log_date) OVER (
            PARTITION BY msno
            ORDER BY log_date
        ) AS prev_log_date

    FROM daily_activity
),

-- Flag the start of a new session (gap > 1 day from previous active day)
session_flags AS (
    SELECT
        *,
        CASE
            WHEN prev_log_date IS NULL THEN 1               -- first ever activity
            WHEN DATE_DIFF('day', prev_log_date, log_date) > 1 THEN 1  -- gap detected
            ELSE 0
        END AS is_session_start

    FROM with_prev_date
),

-- Assign session IDs using a running sum of session start flags
session_ids AS (
    SELECT
        *,
        SUM(is_session_start) OVER (
            PARTITION BY msno
            ORDER BY log_date
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS session_id

    FROM session_flags
)

-- Aggregate to session level
SELECT
    msno,
    session_id,
    MIN(log_date)           AS session_start,
    MAX(log_date)           AS session_end,
    COUNT(*)                AS session_length_days,
    SUM(daily_hours)        AS session_total_hours,
    AVG(daily_tracks)       AS session_avg_daily_tracks,
    SUM(daily_tracks)       AS session_total_tracks

FROM session_ids
GROUP BY msno, session_id;
