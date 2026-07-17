-- Illustrative delayed-outcome monitoring view for booked-credit cohorts.
-- Outcome metrics are reported only when the configured performance window matures.

WITH scored AS (
    SELECT
        s.application_id,
        s.model_version,
        s.scored_at,
        s.probability_bad,
        s.decision,
        s.reason_code_1,
        s.reason_code_2,
        DATEADD(day, -DATEPART(weekday, CAST(s.scored_at AS date)) + 1,
                CAST(s.scored_at AS date)) AS score_week
    FROM dbo.credit_scores AS s
    WHERE s.scored_at >= @window_start
      AND s.scored_at < @window_end
),
outcomes AS (
    SELECT
        o.application_id,
        o.bad_within_performance_window,
        o.label_available_at
    FROM dbo.credit_performance_labels AS o
),
cohorts AS (
    SELECT
        s.score_week,
        s.model_version,
        COUNT(*) AS applications,
        AVG(CAST(CASE WHEN s.decision = 'approve' THEN 1 ELSE 0 END AS float)) AS approval_rate,
        AVG(s.probability_bad) AS mean_probability_bad,
        SUM(CASE WHEN o.label_available_at <= @as_of_ts THEN 1 ELSE 0 END) AS matured_labels,
        AVG(CASE
            WHEN o.label_available_at <= @as_of_ts
            THEN CAST(o.bad_within_performance_window AS float)
        END) AS realized_bad_rate,
        AVG(CASE
            WHEN o.label_available_at <= @as_of_ts
            THEN POWER(s.probability_bad - o.bad_within_performance_window, 2)
        END) AS brier_score
    FROM scored AS s
    LEFT JOIN outcomes AS o ON o.application_id = s.application_id
    GROUP BY s.score_week, s.model_version
)
SELECT
    score_week,
    model_version,
    applications,
    approval_rate,
    mean_probability_bad,
    matured_labels,
    realized_bad_rate,
    brier_score
FROM cohorts
ORDER BY score_week, model_version;
