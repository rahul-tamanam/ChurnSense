-- =============================================================================
-- stg_user_logs.sql
-- Clean and type-cast the raw user_logs table.
--
-- Source columns:
--   msno, date, num_25, num_50, num_75, num_985, num_100, num_unq, total_secs
--
-- num_XX = songs listened to at least XX% completion
-- num_unq = unique songs
-- total_secs = total listening seconds
--
-- Key transforms:
--   - Parse date (integer YYYYMMDD → DATE)
--   - Derive completion rate (fully listened / total played)
--   - Derive skip rate (songs <25% completion / total)
--   - Total tracks played = sum of all completion buckets
--   - Listening hours (total_secs / 3600)
-- =============================================================================

CREATE OR REPLACE TABLE stg_user_logs AS

WITH parsed AS (
    SELECT
        msno,

        TRY_CAST(
            STRPTIME(CAST(date AS VARCHAR), '%Y%m%d')
        AS DATE) AS log_date,

        -- Completion buckets (already integers in source)
        COALESCE(num_25,  0) AS num_25,
        COALESCE(num_50,  0) AS num_50,
        COALESCE(num_75,  0) AS num_75,
        COALESCE(num_985, 0) AS num_985,
        COALESCE(num_100, 0) AS num_100,
        COALESCE(num_unq, 0) AS num_unq,
        COALESCE(total_secs, 0) AS total_secs

    FROM raw_user_logs
    WHERE msno IS NOT NULL
),

with_derived AS (
    SELECT
        *,

        -- Total tracks attempted (all buckets)
        (num_25 + num_50 + num_75 + num_985 + num_100) AS total_tracks,

        -- Listening hours
        ROUND(total_secs / 3600.0, 4) AS listening_hours,

        -- Completion rate: tracks fully heard / total tracks
        CASE
            WHEN (num_25 + num_50 + num_75 + num_985 + num_100) > 0
                THEN ROUND(
                    num_100::FLOAT /
                    (num_25 + num_50 + num_75 + num_985 + num_100), 4)
            ELSE NULL
        END AS completion_rate,

        -- Skip rate: tracks barely started (<25%) / total tracks
        CASE
            WHEN (num_25 + num_50 + num_75 + num_985 + num_100) > 0
                THEN ROUND(
                    num_25::FLOAT /
                    (num_25 + num_50 + num_75 + num_985 + num_100), 4)
            ELSE NULL
        END AS skip_rate

    FROM parsed
    WHERE log_date IS NOT NULL
)

SELECT * FROM with_derived;
