"""Global and local model-behavior explanations without causal claims."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance


def global_permutation_importance(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    n_repeats: int = 30,
) -> pd.DataFrame:
    result = permutation_importance(
        model,
        X,
        y,
        scoring="roc_auc",
        n_repeats=n_repeats,
        random_state=42,
        n_jobs=1,
    )
    return (
        pd.DataFrame(
            {
                "feature": X.columns,
                "mean_auc_decrease": result.importances_mean,
                "std_auc_decrease": result.importances_std,
            }
        )
        .sort_values("mean_auc_decrease", ascending=False)
        .reset_index(drop=True)
    )


def reference_values(frame: pd.DataFrame, categorical_features: list[str]) -> Mapping[str, float]:
    values: dict[str, float] = {}
    for column in frame.columns:
        if column in categorical_features:
            values[column] = float(frame[column].mode().iloc[0])
        else:
            values[column] = float(frame[column].median())
    return values


def local_reason_codes(
    model,
    X: pd.DataFrame,
    *,
    application_ids: pd.Series,
    reference: Mapping[str, float],
    top_n: int = 4,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    base_probability = model.predict_proba(X)[:, 1]
    for row_position, (_, applicant) in enumerate(X.iterrows()):
        impacts = []
        for feature in X.columns:
            counterfactual = applicant.to_frame().T.copy()
            counterfactual[feature] = reference[feature]
            reference_probability = float(model.predict_proba(counterfactual)[:, 1][0])
            impacts.append((feature, float(base_probability[row_position] - reference_probability)))
        for rank, (feature, delta) in enumerate(
            sorted(impacts, key=lambda item: item[1], reverse=True)[:top_n], start=1
        ):
            rows.append(
                {
                    "application_id": int(application_ids.iloc[row_position]),
                    "rank": rank,
                    "feature": feature,
                    "sample_probability_bad": float(base_probability[row_position]),
                    "probability_delta_vs_reference": delta,
                }
            )
    return pd.DataFrame(rows)
