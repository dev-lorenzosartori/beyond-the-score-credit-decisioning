"""Population-stability and score-monitoring helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _psi_from_shares(reference_share: np.ndarray, current_share: np.ndarray) -> float:
    reference = np.clip(reference_share.astype(float), 1e-6, None)
    current = np.clip(current_share.astype(float), 1e-6, None)
    reference /= reference.sum()
    current /= current.sum()
    return float(np.sum((current - reference) * np.log(current / reference)))


def numeric_psi(reference: pd.Series, current: pd.Series, *, bins: int = 10) -> float:
    quantiles = np.unique(reference.quantile(np.linspace(0, 1, bins + 1)).to_numpy())
    if len(quantiles) < 3:
        return categorical_psi(reference.astype(str), current.astype(str))
    quantiles[0], quantiles[-1] = -np.inf, np.inf
    reference_bins = pd.cut(reference, bins=quantiles, include_lowest=True)
    current_bins = pd.cut(current, bins=quantiles, include_lowest=True)
    categories = reference_bins.cat.categories
    reference_share = reference_bins.value_counts(sort=False).reindex(categories, fill_value=0).to_numpy()
    current_share = current_bins.value_counts(sort=False).reindex(categories, fill_value=0).to_numpy()
    return _psi_from_shares(reference_share, current_share)


def categorical_psi(reference: pd.Series, current: pd.Series) -> float:
    categories = sorted(set(reference.dropna().astype(str)) | set(current.dropna().astype(str)))
    reference_share = reference.astype(str).value_counts().reindex(categories, fill_value=0).to_numpy()
    current_share = current.astype(str).value_counts().reindex(categories, fill_value=0).to_numpy()
    return _psi_from_shares(reference_share, current_share)


def drift_severity(psi: float) -> str:
    if psi < 0.10:
        return "stable"
    if psi < 0.25:
        return "watch"
    return "action"


def population_stability_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    *,
    numeric_features: list[str],
) -> pd.DataFrame:
    rows = []
    for feature in reference.columns:
        psi = (
            numeric_psi(reference[feature], current[feature])
            if feature in numeric_features
            else categorical_psi(reference[feature], current[feature])
        )
        rows.append({"feature": feature, "psi": psi, "severity": drift_severity(psi)})
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)


def simulate_drift_batch(frame: pd.DataFrame) -> pd.DataFrame:
    """Create a deterministic demonstration batch; never presented as observed production data."""
    current = frame.copy()
    current["amount"] = np.rint(current.amount * 1.25).astype(int)
    current["duration"] = np.minimum(current.duration + 6, 72)
    shift_count = max(1, int(0.20 * len(current)))
    shift_index = current.sort_values("application_id").index[:shift_count]
    current.loc[shift_index, "status"] = 1
    return current
