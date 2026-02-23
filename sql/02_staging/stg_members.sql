-- =============================================================================
-- stg_members.sql
-- Clean and type-cast the raw members table.
--
-- Key transforms:
--   - Parse registration_init_time (integer YYYYMMDD → DATE)
--   - Bucket age into clean cohorts (raw has outliers: age=0, age=2000)
--   - Encode gender as integer flag
--   - Derive tenure_days from registration to reference date 2017-03-31
-- =============================================================================

CREATE OR REPLACE TABLE stg_members AS

WITH base AS (
    SELECT
        msno,

        -- Gender: clean to binary flag, NULL for unknown
        CASE
            WHEN gender = 'male'   THEN 1
            WHEN gender = 'female' THEN 0
            ELSE NULL
        END AS gender_male_flag,

        city,
        registered_via,

        -- Registration date: stored as integer YYYYMMDD
        TRY_CAST(
            STRPTIME(CAST(registration_init_time AS VARCHAR), '%Y%m%d')
        AS DATE) AS registration_date,

        -- Age: raw has extreme outliers (0, 1, >100, >1000)
        -- Anything outside 10-75 treated as unknown
        CASE
            WHEN bd BETWEEN 10 AND 75 THEN bd
            ELSE NULL
        END AS age

    FROM raw_members
),

with_derived AS (
    SELECT
        *,

        CASE
            WHEN age IS NULL THEN 'unknown'
            WHEN age < 20    THEN '10-19'
            WHEN age < 30    THEN '20-29'
            WHEN age < 40    THEN '30-39'
            WHEN age < 50    THEN '40-49'
            ELSE                  '50+'
        END AS age_bucket,

        -- Tenure: days from registration to KKBox prediction reference date
        DATE_DIFF('day', registration_date, DATE '2017-03-31') AS tenure_days

    FROM base
)

SELECT * FROM with_derived
WHERE msno IS NOT NULL;
