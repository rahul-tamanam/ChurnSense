CREATE OR REPLACE TABLE feat_usage_decay AS

WITH ref AS (
    SELECT DATE '2017-03-31' AS ref_date
),

-- Single scan — pull last 90 days only, tag each row with days_ago
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

-- Everything computed in ONE GROUP BY — no repeated scans
agg AS (
    SELECT
        msno,

        -- 30d
        COUNT(DISTINCT CASE WHEN days_ago <= 30 THEN log_date END)  AS sessions_last_30d,
        SUM(CASE WHEN days_ago <= 30 THEN listening_hours ELSE 0 END) AS hours_last_30d,
        SUM(CASE WHEN days_ago <= 30 THEN total_tracks    ELSE 0 END) AS tracks_last_30d,
        AVG(CASE WHEN days_ago <= 30 THEN listening_hours END)       AS avg_daily_hours_last_30d,
        AVG(CASE WHEN days_ago <= 30 THEN completion_rate END)       AS completion_rate_avg_30d,
        AVG(CASE WHEN days_ago <= 30 THEN skip_rate       END)       AS skip_rate_avg_30d,

        -- 60d
        COUNT(DISTINCT CASE WHEN days_ago <= 60 THEN log_date END)  AS sessions_last_60d,
        SUM(CASE WHEN days_ago <= 60 THEN listening_hours ELSE 0 END) AS hours_last_60d,
        SUM(CASE WHEN days_ago <= 60 THEN total_tracks    ELSE 0 END) AS tracks_last_60d,
        AVG(CASE WHEN days_ago <= 60 THEN listening_hours END)       AS avg_daily_hours_last_60d,

        -- 90d
        COUNT(DISTINCT CASE WHEN days_ago <= 90 THEN log_date END)  AS sessions_last_90d,
        SUM(CASE WHEN days_ago <= 90 THEN listening_hours ELSE 0 END) AS hours_last_90d,
        SUM(CASE WHEN days_ago <= 90 THEN total_tracks    ELSE 0 END) AS tracks_last_90d,
        AVG(CASE WHEN days_ago <= 90 THEN listening_hours END)       AS avg_daily_hours_last_90d,

        -- Decay: first half vs second half of each window
        -- 30d decay (days 1-15 vs 16-30)
        ROUND(
            (AVG(CASE WHEN days_ago BETWEEN 16 AND 30 THEN listening_hours END) -
             AVG(CASE WHEN days_ago BETWEEN  1 AND 15 THEN listening_hours END))
            / NULLIF(AVG(CASE WHEN days_ago BETWEEN 16 AND 30 THEN listening_hours END), 0)
        , 4) AS usage_decay_30d,

        -- 60d decay (days 1-30 vs 31-60)
        ROUND(
            (AVG(CASE WHEN days_ago BETWEEN 31 AND 60 THEN listening_hours END) -
             AVG(CASE WHEN days_ago BETWEEN  1 AND 30 THEN listening_hours END))
            / NULLIF(AVG(CASE WHEN days_ago BETWEEN 31 AND 60 THEN listening_hours END), 0)
        , 4) AS usage_decay_60d,

        -- 90d decay (days 1-45 vs 46-90)
        ROUND(
            (AVG(CASE WHEN days_ago BETWEEN 46 AND 90 THEN listening_hours END) -
             AVG(CASE WHEN days_ago BETWEEN  1 AND 45 THEN listening_hours END))
            / NULLIF(AVG(CASE WHEN days_ago BETWEEN 46 AND 90 THEN listening_hours END), 0)
        , 4) AS usage_decay_90d

    FROM windowed
    GROUP BY msno
)

SELECT * FROM agg;