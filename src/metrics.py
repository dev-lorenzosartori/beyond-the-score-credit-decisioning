"""Credit-risk metrics with explicit probability and risk conventions."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
    roc_curve,
)


def prior_probability_shift(
    probability: Sequence[float], source_rate: float, target_rate: float
) -> np.ndarray:
    """Adjust calibrated sample probabilities for a target population prior."""
    if not 0 < source_rate < 1 or not 0 < target_rate < 1:
        raise ValueError("Source and target rates must be between zero and one")
    p = np.clip(np.asarray(probability, dtype=float), 1e-9, 1 - 1e-9)
    sample_odds = p / (1 - p)
    odds_multiplier = (target_rate / (1 - target_rate)) / (source_rate / (1 - source_rate))
    population_odds = sample_odds * odds_multiplier
    return population_odds / (1 + population_odds)


def ks_statistic(y_true: Sequence[int], probability_bad: Sequence[float]) -> float:
    fpr, tpr, _ = roc_curve(y_true, probability_bad)
    return float(np.max(tpr - fpr))


def bootstrap_auc_interval(
    y_true: Sequence[int], probability_bad: Sequence[float], *, n_bootstrap: int = 1000
) -> tuple[float, float]:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probability_bad, dtype=float)
    rng = np.random.default_rng(42)
    estimates: list[float] = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(y), len(y))
        if np.unique(y[idx]).size < 2:
            continue
        estimates.append(float(roc_auc_score(y[idx], p[idx])))
    if not estimates:
        raise ValueError("No valid bootstrap samples")
    return tuple(np.quantile(estimates, [0.025, 0.975]))


def evaluate_probabilities(
    y_true: Sequence[int],
    probability_bad: Sequence[float],
    *,
    model: str,
    split: str,
    n_bootstrap: int = 1000,
) -> dict[str, float | str]:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probability_bad, dtype=float)
    auc = float(roc_auc_score(y, p))
    low, high = bootstrap_auc_interval(y, p, n_bootstrap=n_bootstrap)
    return {
        "model": model,
        "split": split,
        "roc_auc": auc,
        "roc_auc_ci_low": low,
        "roc_auc_ci_high": high,
        "gini": 2 * auc - 1,
        "average_precision": float(average_precision_score(y, p)),
        "brier_score": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p)),
        "ks": ks_statistic(y, p),
        "mean_probability_bad": float(p.mean()),
    }


def calibration_table(
    y_true: Sequence[int], probability_bad: Sequence[float], *, bins: int = 5
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {"observed_bad": np.asarray(y_true, dtype=int), "predicted_bad": probability_bad}
    )
    frame["bin"] = pd.qcut(frame.predicted_bad, q=bins, labels=False, duplicates="drop")
    return (
        frame.groupby("bin", as_index=False, observed=True)
        .agg(
            n=("observed_bad", "size"),
            mean_predicted_bad=("predicted_bad", "mean"),
            observed_bad_rate=("observed_bad", "mean"),
        )
        .sort_values("mean_predicted_bad")
    )
