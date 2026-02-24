-- =============================================================================
-- feat_login_frequency.sql
-- Login cadence, gap, and streak features derived from sessions.
--
-- Built on top of feat_sessions (sessionization output).
-- =============================================================================

CREATE OR REPLACE TABLE feat_login_frequency AS

WITH ref AS (
    SELECT DATE '2017-03-31' AS ref_date
),

-- Use sessions table — each row is a streak of consecutive active days
session_gaps AS (
    SELECT
        msno,
        session_id,
        session_start,
        session_end,
        session_length_days,

        DATE_DIFF(
            'day',
            LAG(session_end) OVER (PARTITION BY msno ORDER BY session_id),
            session_start
        ) AS gap_before_session

    FROM feat_sessions
),

-- Per-user aggregates over all sessions
user_session_stats AS (
    SELECT
        msno,
        MAX(session_end)         AS last_active_date,
        MAX(session_length_days) AS login_streak_max,
        AVG(gap_before_session)  AS login_gap_days_avg,
        MAX(gap_before_session)  AS login_gap_days_max
    FROM session_gaps
    GROUP BY msno
),

-- Most recent gap (clean window logic)
recent_gap AS (
    SELECT
        msno,
        gap_before_session AS login_gap_days_recent
    FROM (
        SELECT
            msno,
            gap_before_session,
            ROW_NUMBER() OVER (
                PARTITION BY msno
                ORDER BY session_id DESC
            ) AS rn
        FROM session_gaps
    ) t
    WHERE rn = 1
),

-- Active day counts per window
windowed_logins AS (
    SELECT
        l.msno,

        COUNT(DISTINCT CASE
            WHEN DATE_DIFF('day', l.log_date, r.ref_date) <= 30
            THEN l.log_date END
        ) AS login_days_last_30d,

        COUNT(DISTINCT CASE
            WHEN DATE_DIFF('day', l.log_date, r.ref_date) <= 60
            THEN l.log_date END
        ) AS login_days_last_60d

    FROM stg_user_logs l
    CROSS JOIN ref r
    WHERE l.log_date >= (r.ref_date - INTERVAL '60 days')
    GROUP BY l.msno
),

-- Current streak approximation
current_streak AS (
    SELECT
        msno,
        CASE
            WHEN DATE_DIFF('day', session_end, ref_date) <= 3
                THEN session_length_days
            ELSE 0
        END AS login_streak_current
    FROM (
        SELECT
            s.msno,
            s.session_end,
            s.session_length_days,
            r.ref_date,
            ROW_NUMBER() OVER (
                PARTITION BY s.msno
                ORDER BY s.session_id DESC
            ) AS rn
        FROM feat_sessions s
        CROSS JOIN ref r
    ) t
    WHERE rn = 1
)

SELECT
    u.msno,
    COALESCE(w.login_days_last_30d, 0)              AS login_days_last_30d,
    COALESCE(w.login_days_last_60d, 0)              AS login_days_last_60d,
    ROUND(COALESCE(u.login_gap_days_avg, 0), 2)     AS login_gap_days_avg,
    COALESCE(u.login_gap_days_max, 0)               AS login_gap_days_max,
    COALESCE(rg.login_gap_days_recent, 0)           AS login_gap_days_recent,
    COALESCE(u.login_streak_max, 0)                 AS login_streak_max,
    COALESCE(c.login_streak_current, 0)             AS login_streak_current,
    DATE_DIFF('day', u.last_active_date, ref.ref_date)
        AS days_since_last_login

FROM user_session_stats u
LEFT JOIN windowed_logins w USING (msno)
LEFT JOIN current_streak c USING (msno)
LEFT JOIN recent_gap rg USING (msno)
CROSS JOIN ref;