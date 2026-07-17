"""Audit decision outcomes for the limited groups available in the dataset."""

from __future__ import annotations

from math import sqrt

import numpy as np
import pandas as pd


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total == 0:
        return np.nan, np.nan
    p = successes / total
    denominator = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denominator
    spread = z * sqrt((p * (1 - p) + z**2 / (4 * total)) / total) / denominator
    return center - spread, center + spread


def _group_rows(frame: pd.DataFrame, group_column: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for group, segment in frame.groupby(group_column, observed=True):
        good = segment.observed_bad.eq(0)
        bad = segment.observed_bad.eq(1)
        selected = int(segment.approved.sum())
        low, high = wilson_interval(selected, len(segment))
        rows.append(
            {
                "audit_dimension": group_column,
                "group": str(group),
                "n": len(segment),
                "good_n": int(good.sum()),
                "bad_n": int(bad.sum()),
                "selection_rate": float(segment.approved.mean()),
                "selection_rate_ci_low": low,
                "selection_rate_ci_high": high,
                "good_applicant_approval_rate": float(segment.loc[good, "approved"].mean()),
                "bad_applicant_approval_rate": float(segment.loc[bad, "approved"].mean()),
                "observed_bad_rate": float(segment.observed_bad.mean()),
                "mean_adjusted_probability_bad": float(segment.adjusted_probability_bad.mean()),
                "small_sample_warning": len(segment) < 30 or good.sum() < 20 or bad.sum() < 20,
            }
        )
    return rows


def fairness_audit(
    scored_frame: pd.DataFrame,
    approved: np.ndarray,
    adjusted_probability_bad: np.ndarray,
) -> pd.DataFrame:
    audit = scored_frame[["age", "foreign_worker", "bad_credit"]].copy()
    audit["approved"] = np.asarray(approved, dtype=bool)
    audit["adjusted_probability_bad"] = np.asarray(adjusted_probability_bad, dtype=float)
    audit = audit.rename(columns={"bad_credit": "observed_bad"})
    audit["age_group"] = np.where(audit.age < 25, "under_25", "25_plus")
    audit["worker_group"] = audit.foreign_worker.map(
        {1: "foreign_worker", 2: "not_foreign_worker"}
    )
    rows = _group_rows(audit, "age_group") + _group_rows(audit, "worker_group")
    return pd.DataFrame(rows)
