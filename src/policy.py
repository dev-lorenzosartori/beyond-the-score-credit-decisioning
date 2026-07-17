"""Translate probability of default into approval and value decisions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from src.config import (
    FIXED_PD_CUTOFF,
    GOOD_MARGIN_RATE,
    LOSS_GIVEN_DEFAULT,
    OPERATING_COST,
    SOURCE_BAD_RATE,
    TARGET_BAD_RATE,
)


@dataclass(frozen=True)
class EconomicAssumptions:
    target_bad_rate: float = TARGET_BAD_RATE
    good_margin_rate: float = GOOD_MARGIN_RATE
    loss_given_default: float = LOSS_GIVEN_DEFAULT
    operating_cost: float = OPERATING_COST
    fixed_pd_cutoff: float = FIXED_PD_CUTOFF


def expected_value(
    probability_bad: np.ndarray,
    amount_proxy: np.ndarray,
    assumptions: EconomicAssumptions = EconomicAssumptions(),
) -> np.ndarray:
    p = np.asarray(probability_bad, dtype=float)
    amount = np.asarray(amount_proxy, dtype=float)
    return (
        (1 - p) * assumptions.good_margin_rate * amount
        - p * assumptions.loss_given_default * amount
        - assumptions.operating_cost
    )


def policy_decisions(
    probability_bad: np.ndarray,
    amount_proxy: np.ndarray,
    assumptions: EconomicAssumptions = EconomicAssumptions(),
) -> Mapping[str, np.ndarray]:
    p = np.asarray(probability_bad, dtype=float)
    return {
        "Approve all": np.ones(len(p), dtype=bool),
        "Fixed PD cutoff": p < assumptions.fixed_pd_cutoff,
        "Expected-value policy": expected_value(p, amount_proxy, assumptions) > 0,
    }


def population_weights(
    observed_bad: np.ndarray,
    *,
    source_bad_rate: float = SOURCE_BAD_RATE,
    target_bad_rate: float = TARGET_BAD_RATE,
) -> np.ndarray:
    y = np.asarray(observed_bad, dtype=int)
    return np.where(
        y == 1,
        target_bad_rate / source_bad_rate,
        (1 - target_bad_rate) / (1 - source_bad_rate),
    )


def realized_value(
    observed_bad: np.ndarray,
    amount_proxy: np.ndarray,
    assumptions: EconomicAssumptions = EconomicAssumptions(),
) -> np.ndarray:
    y = np.asarray(observed_bad, dtype=int)
    amount = np.asarray(amount_proxy, dtype=float)
    return np.where(
        y == 0,
        assumptions.good_margin_rate * amount - assumptions.operating_cost,
        -assumptions.loss_given_default * amount - assumptions.operating_cost,
    )


def evaluate_policies(
    observed_bad: np.ndarray,
    adjusted_probability_bad: np.ndarray,
    amount_proxy: np.ndarray,
    *,
    model: str,
    split: str,
    assumptions: EconomicAssumptions = EconomicAssumptions(),
) -> pd.DataFrame:
    y = np.asarray(observed_bad, dtype=int)
    p = np.asarray(adjusted_probability_bad, dtype=float)
    amount = np.asarray(amount_proxy, dtype=float)
    weights = population_weights(y, target_bad_rate=assumptions.target_bad_rate)
    outcomes = realized_value(y, amount, assumptions)
    rows = []
    for policy, approved in policy_decisions(p, amount, assumptions).items():
        approved_weight = weights[approved].sum()
        bad_rate = (
            np.average(y[approved], weights=weights[approved]) if approved_weight else np.nan
        )
        rows.append(
            {
                "model": model,
                "split": split,
                "policy": policy,
                "approval_rate": float(np.average(approved, weights=weights)),
                "expected_bad_rate": float(bad_rate),
                "value_units_per_100_applications": float(
                    100 * np.sum(outcomes[approved] * weights[approved]) / weights.sum()
                ),
                "approved_rows": int(approved.sum()),
            }
        )
    return pd.DataFrame(rows)


def bootstrap_policy_value_interval(
    observed_bad: np.ndarray,
    approved: np.ndarray,
    amount_proxy: np.ndarray,
    *,
    n_bootstrap: int = 1000,
    assumptions: EconomicAssumptions = EconomicAssumptions(),
) -> tuple[float, float]:
    y = np.asarray(observed_bad, dtype=int)
    approved = np.asarray(approved, dtype=bool)
    amount = np.asarray(amount_proxy, dtype=float)
    rng = np.random.default_rng(42)
    estimates = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(y), len(y))
        weights = population_weights(y[idx], target_bad_rate=assumptions.target_bad_rate)
        outcomes = realized_value(y[idx], amount[idx], assumptions)
        estimates.append(
            100
            * np.sum(outcomes[approved[idx]] * weights[approved[idx]])
            / weights.sum()
        )
    return tuple(np.quantile(estimates, [0.025, 0.975]))
