-- Illustrative Microsoft Fabric / T-SQL feature mart.
-- Goal: compute application-time features without using post-decision information.

WITH application_base AS (
    SELECT
        a.application_id,
        a.customer_id,
        a.application_ts,
        a.requested_amount,
        a.requested_term_months,
        a.product_code
    FROM dbo.credit_applications AS a
    WHERE a.application_ts >= @window_start
      AND a.application_ts < @window_end
),
account_history AS (
    SELECT
        a.application_id,
        COUNT(DISTINCT h.account_id) AS prior_account_count,
        SUM(CASE WHEN h.days_past_due >= 30 THEN 1 ELSE 0 END) AS prior_30dpd_count,
        MAX(h.days_past_due) AS max_prior_days_past_due,
        SUM(h.current_balance) AS prior_balance
    FROM application_base AS a
    LEFT JOIN dbo.customer_account_daily AS h
      ON h.customer_id = a.customer_id
     AND h.snapshot_date < CAST(a.application_ts AS date)
     AND h.snapshot_date >= DATEADD(month, -12, CAST(a.application_ts AS date))
    GROUP BY a.application_id
),
payment_history AS (
    SELECT
        a.application_id,
        COUNT(p.payment_id) AS payments_last_180d,
        SUM(CASE WHEN p.days_late > 0 THEN 1 ELSE 0 END) AS late_payments_last_180d,
        AVG(CAST(p.days_late AS float)) AS avg_days_late_last_180d
    FROM application_base AS a
    LEFT JOIN dbo.payment_events AS p
      ON p.customer_id = a.customer_id
     AND p.payment_ts < a.application_ts
     AND p.payment_ts >= DATEADD(day, -180, a.application_ts)
    GROUP BY a.application_id
)
SELECT
    a.application_id,
    a.application_ts,
    a.requested_amount,
    a.requested_term_months,
    a.product_code,
    COALESCE(h.prior_account_count, 0) AS prior_account_count,
    COALESCE(h.prior_30dpd_count, 0) AS prior_30dpd_count,
    COALESCE(h.max_prior_days_past_due, 0) AS max_prior_days_past_due,
    COALESCE(h.prior_balance, 0) AS prior_balance,
    COALESCE(p.payments_last_180d, 0) AS payments_last_180d,
    COALESCE(p.late_payments_last_180d, 0) AS late_payments_last_180d,
    COALESCE(p.avg_days_late_last_180d, 0) AS avg_days_late_last_180d
FROM application_base AS a
LEFT JOIN account_history AS h ON h.application_id = a.application_id
LEFT JOIN payment_history AS p ON p.application_id = a.application_id;
