"""Train, validate and externally report the Beyond the Score case."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    CATEGORICAL_FEATURES,
    FIGURES_DIR,
    MODEL_FEATURES,
    MODELS_DIR,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    REPORTS_DIR,
    SOURCE_BAD_RATE,
    TARGET_BAD_RATE,
)
from src.data import data_quality_summary, load_credit_data, model_xy, split_credit_data
from src.explainability import global_permutation_importance, local_reason_codes, reference_values
from src.fairness import fairness_audit
from src.metrics import calibration_table, evaluate_probabilities, prior_probability_shift
from src.monitoring import population_stability_report, simulate_drift_batch
from src.policy import (
    EconomicAssumptions,
    bootstrap_policy_value_interval,
    evaluate_policies,
    expected_value,
    policy_decisions,
)

BLUE = "#2563EB"
GOLD = "#D4A017"
ORANGE = "#F97316"
PURPLE = "#7C3AED"
SLATE = "#64748B"
LIGHT_SLATE = "#CBD5E1"
INK = "#172033"


def preprocessing_pipeline() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def build_models() -> dict[str, object]:
    logistic = Pipeline(
        [
            ("preprocessing", preprocessing_pipeline()),
            (
                "model",
                LogisticRegression(C=0.5, max_iter=2000, random_state=RANDOM_STATE),
            ),
        ]
    )
    gradient_base = Pipeline(
        [
            ("preprocessing", preprocessing_pipeline()),
            (
                "model",
                GradientBoostingClassifier(
                    n_estimators=150,
                    learning_rate=0.03,
                    max_depth=2,
                    min_samples_leaf=12,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    calibration_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    gradient = CalibratedClassifierCV(
        gradient_base,
        method="sigmoid",
        cv=calibration_cv,
        n_jobs=1,
    )
    return {"Logistic Regression": logistic, "Calibrated Gradient Boosting": gradient}


def prepare_directories() -> None:
    for directory in (REPORTS_DIR, FIGURES_DIR, MODELS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def evaluate(*, n_bootstrap: int = 1000) -> dict[str, object]:
    prepare_directories()
    frame = load_credit_data()
    splits = split_credit_data(frame)
    X_train, y_train = model_xy(splits.train)

    fitted: dict[str, object] = {}
    model_rows: list[dict[str, float | str]] = []
    policy_tables: list[pd.DataFrame] = []
    probability_cache: dict[tuple[str, str], np.ndarray] = {}
    split_frames = {"validation": splits.validation, "test": splits.test}

    for model_name, model in build_models().items():
        model.fit(X_train, y_train)
        fitted[model_name] = model
        for split_name, split_frame in split_frames.items():
            X_split, y_split = model_xy(split_frame)
            sample_probability = model.predict_proba(X_split)[:, 1]
            probability_cache[(model_name, split_name)] = sample_probability
            model_rows.append(
                evaluate_probabilities(
                    y_split,
                    sample_probability,
                    model=model_name,
                    split=split_name,
                    n_bootstrap=n_bootstrap,
                )
            )
            adjusted_probability = prior_probability_shift(
                sample_probability, SOURCE_BAD_RATE, TARGET_BAD_RATE
            )
            policy_tables.append(
                evaluate_policies(
                    y_split.to_numpy(),
                    adjusted_probability,
                    split_frame.amount.to_numpy(),
                    model=model_name,
                    split=split_name,
                )
            )

    metrics = pd.DataFrame(model_rows)
    policies = pd.concat(policy_tables, ignore_index=True)
    champion = select_champion(policies)
    champion_model = fitted[champion]
    joblib.dump(champion_model, MODELS_DIR / "champion_credit_decisioning.joblib")

    X_test, y_test = model_xy(splits.test)
    champion_sample_probability = probability_cache[(champion, "test")]
    champion_adjusted_probability = prior_probability_shift(
        champion_sample_probability, SOURCE_BAD_RATE, TARGET_BAD_RATE
    )
    assumptions = EconomicAssumptions()
    decisions = policy_decisions(
        champion_adjusted_probability, splits.test.amount.to_numpy(), assumptions
    )["Expected-value policy"]
    expected_values = expected_value(
        champion_adjusted_probability, splits.test.amount.to_numpy(), assumptions
    )

    fairness = fairness_audit(splits.test, decisions, champion_adjusted_probability)
    importance = global_permutation_importance(champion_model, X_test, y_test)
    reasons = local_reason_codes(
        champion_model,
        X_test.iloc[np.argsort(champion_adjusted_probability)[-10:][::-1]],
        application_ids=splits.test.application_id.iloc[
            np.argsort(champion_adjusted_probability)[-10:][::-1]
        ].reset_index(drop=True),
        reference=reference_values(X_train, CATEGORICAL_FEATURES),
    )

    calibrations = []
    for model_name in fitted:
        calibrations.append(
            calibration_table(
                y_test,
                probability_cache[(model_name, "test")],
                bins=5,
            ).assign(model=model_name)
        )
    calibration = pd.concat(calibrations, ignore_index=True)

    test_decisions = splits.test[["application_id", "amount", "bad_credit"]].copy()
    test_decisions["sample_probability_bad"] = champion_sample_probability
    test_decisions["adjusted_probability_bad"] = champion_adjusted_probability
    test_decisions["expected_value_units"] = expected_values
    test_decisions["decision"] = np.where(decisions, "approve", "decline")

    drift_batch = simulate_drift_batch(splits.test)
    drift_X, _ = model_xy(drift_batch)
    reference_monitoring = X_train.copy()
    current_monitoring = drift_X.copy()
    reference_monitoring["score_probability_bad"] = champion_model.predict_proba(X_train)[:, 1]
    current_monitoring["score_probability_bad"] = champion_model.predict_proba(drift_X)[:, 1]
    monitoring = population_stability_report(
        reference_monitoring,
        current_monitoring,
        numeric_features=[*NUMERIC_FEATURES, "score_probability_bad"],
    )

    expected_policy = policies[
        (policies.model == champion)
        & (policies.split == "test")
        & (policies.policy == "Expected-value policy")
    ].iloc[0]
    value_low, value_high = bootstrap_policy_value_interval(
        y_test.to_numpy(),
        decisions,
        splits.test.amount.to_numpy(),
        n_bootstrap=n_bootstrap,
    )

    data_quality_summary(frame).to_csv(REPORTS_DIR / "data_quality_checks.csv", index=False)
    metrics.to_csv(REPORTS_DIR / "model_metrics.csv", index=False)
    policies.to_csv(REPORTS_DIR / "policy_comparison.csv", index=False)
    calibration.to_csv(REPORTS_DIR / "calibration_test.csv", index=False)
    fairness.to_csv(REPORTS_DIR / "fairness_audit.csv", index=False)
    importance.to_csv(REPORTS_DIR / "feature_importance.csv", index=False)
    reasons.to_csv(REPORTS_DIR / "reason_codes_sample.csv", index=False)
    test_decisions.to_csv(REPORTS_DIR / "test_decisions.csv", index=False)
    monitoring.to_csv(REPORTS_DIR / "monitoring_demo.csv", index=False)

    summary = {
        "champion": champion,
        "validation_selection_metric": "value_units_per_100_applications",
        "test_expected_value_policy": {
            "approval_rate": float(expected_policy.approval_rate),
            "expected_bad_rate": float(expected_policy.expected_bad_rate),
            "value_units_per_100_applications": float(
                expected_policy.value_units_per_100_applications
            ),
            "value_95_ci": [float(value_low), float(value_high)],
        },
        "scenario": {
            "source_bad_rate": SOURCE_BAD_RATE,
            "target_bad_rate": TARGET_BAD_RATE,
            "good_margin_rate": assumptions.good_margin_rate,
            "loss_given_default": assumptions.loss_given_default,
            "operating_cost_units": assumptions.operating_cost,
            "amount_warning": "Amount is an historical transformed proxy; values are illustrative, not currency forecasts.",
        },
    }
    (REPORTS_DIR / "decision_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    render_figures(metrics, policies, calibration, fairness, importance, champion)
    write_executive_summary(metrics, policies, fairness, monitoring, champion, summary)
    return summary


def select_champion(policies: pd.DataFrame) -> str:
    candidates = policies[
        (policies.split == "validation") & (policies.policy == "Expected-value policy")
    ].sort_values("value_units_per_100_applications", ascending=False)
    if candidates.empty:
        raise ValueError("No validation policy results available for champion selection")
    return str(candidates.iloc[0].model)


def render_figures(
    metrics: pd.DataFrame,
    policies: pd.DataFrame,
    calibration: pd.DataFrame,
    fairness: pd.DataFrame,
    importance: pd.DataFrame,
    champion: str,
) -> None:
    test_metrics = metrics[metrics.split == "test"].sort_values("roc_auc")
    colors = {
        "Logistic Regression": GOLD,
        "Calibrated Gradient Boosting": BLUE,
    }
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    positions = np.arange(len(test_metrics))
    axes[0].errorbar(
        test_metrics.roc_auc,
        positions,
        xerr=np.vstack(
            [
                test_metrics.roc_auc - test_metrics.roc_auc_ci_low,
                test_metrics.roc_auc_ci_high - test_metrics.roc_auc,
            ]
        ),
        fmt="none",
        ecolor=LIGHT_SLATE,
        capsize=4,
        linewidth=2,
    )
    for position, row in zip(positions, test_metrics.itertuples(), strict=True):
        axes[0].scatter(row.roc_auc, position, s=90, color=colors[row.model], zorder=3)
        axes[0].text(row.roc_auc + 0.012, position, f"{row.roc_auc:.3f}", va="center")
    axes[0].axvline(0.5, color=SLATE, linestyle="--", linewidth=1)
    axes[0].set_yticks(positions, test_metrics.model)
    axes[0].set_xlim(0.45, 0.90)
    axes[0].set_xlabel("ROC-AUC with 95% bootstrap CI")
    axes[0].set_title("Discrimination")
    for position, row in zip(positions, test_metrics.itertuples(), strict=True):
        axes[1].scatter(row.brier_score, position, s=90, color=colors[row.model])
        axes[1].text(row.brier_score + 0.004, position, f"{row.brier_score:.3f}", va="center")
    axes[1].set_yticks(positions, [])
    axes[1].set_xlim(0.12, 0.23)
    axes[1].set_xlabel("Brier score (lower is better)")
    axes[1].set_title("Probability accuracy")
    for ax in axes:
        ax.grid(axis="x", alpha=0.2)
    fig.suptitle("Held-out model comparison", color=INK, fontsize=15)
    fig.text(0.5, 0.01, "South German Credit test partition · n=200", ha="center", color=SLATE)
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    fig.savefig(FIGURES_DIR / "model_comparison.png", dpi=180)
    plt.close(fig)

    champion_policy = policies[(policies.model == champion) & (policies.split == "test")].copy()
    order = ["Approve all", "Fixed PD cutoff", "Expected-value policy"]
    champion_policy["policy"] = pd.Categorical(champion_policy.policy, order, ordered=True)
    champion_policy = champion_policy.sort_values("policy")
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    bars = ax.barh(
        champion_policy.policy.astype(str),
        champion_policy.value_units_per_100_applications,
        color=[LIGHT_SLATE, SLATE, BLUE],
        edgecolor=INK,
        linewidth=0.6,
    )
    for bar, value in zip(bars, champion_policy.value_units_per_100_applications, strict=True):
        ax.text(value + 350, bar.get_y() + bar.get_height() / 2, f"{value:,.0f}", va="center")
    ax.set_xlabel("Illustrative value units per 100 applications")
    fig.suptitle("Test-set policy comparison", y=0.98, fontsize=15, color=INK)
    fig.text(
        0.5,
        0.91,
        "15% target bad-rate scenario · amount treated as a transformed exposure proxy",
        ha="center",
        color=SLATE,
    )
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout(rect=(0, 0, 1, 0.87))
    fig.savefig(FIGURES_DIR / "policy_value_comparison.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    ax.plot([0, 1], [0, 1], color=SLATE, linestyle="--", label="Perfect calibration")
    for model_name, group in calibration.groupby("model"):
        ax.plot(
            group.mean_predicted_bad,
            group.observed_bad_rate,
            marker="o",
            linewidth=2,
            color=colors[model_name],
            label=model_name,
        )
    ax.set(
        xlim=(0, 0.8),
        ylim=(0, 0.8),
        xlabel="Mean predicted bad-credit probability",
        ylabel="Observed bad-credit rate",
    )
    fig.suptitle("Test-set calibration by probability quintile", y=0.98, fontsize=15, color=INK)
    fig.text(0.5, 0.92, "Sample prevalence = 30% · n=200", ha="center", color=SLATE)
    ax.legend(frameon=False, loc="upper left")
    ax.grid(alpha=0.2)
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    fig.savefig(FIGURES_DIR / "calibration.png", dpi=180)
    plt.close(fig)

    age_audit = fairness[fairness.audit_dimension == "age_group"].sort_values("group")
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    x = np.arange(len(age_audit))
    yerr = np.vstack(
        [
            age_audit.selection_rate - age_audit.selection_rate_ci_low,
            age_audit.selection_rate_ci_high - age_audit.selection_rate,
        ]
    )
    ax.bar(
        x,
        age_audit.selection_rate,
        yerr=yerr,
        capsize=5,
        color=["#93B4F7", BLUE],
        edgecolor=INK,
        linewidth=0.6,
    )
    age_labels = {"25_plus": "25+", "under_25": "Under 25"}
    ax.set_xticks(
        x,
        [f"{age_labels[g]}\n(n={n})" for g, n in zip(age_audit.group, age_audit.n, strict=True)],
    )
    ax.set_ylim(0, 1)
    ax.set_ylabel("Raw selection rate")
    fig.suptitle("Age-group decision audit", y=0.98, fontsize=15, color=INK)
    fig.text(
        0.5,
        0.91,
        "Expected-value policy · test set · age excluded from model",
        ha="center",
        color=SLATE,
    )
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout(rect=(0, 0, 1, 0.87))
    fig.savefig(FIGURES_DIR / "fairness_age_audit.png", dpi=180)
    plt.close(fig)

    top = importance.head(10).sort_values("mean_auc_decrease")
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    feature_labels = [name.replace("_", " ").title() for name in top.feature]
    bars = ax.barh(feature_labels, top.mean_auc_decrease, color=BLUE, edgecolor=INK, linewidth=0.5)
    for bar, value in zip(bars, top.mean_auc_decrease, strict=True):
        ax.text(value + 0.002, bar.get_y() + bar.get_height() / 2, f"{value:.3f}", va="center")
    ax.axvline(0, color=SLATE, linewidth=1)
    ax.set_xlabel("Mean ROC-AUC decrease after permutation")
    fig.suptitle("Champion global permutation importance", y=0.98, fontsize=15, color=INK)
    fig.text(0.5, 0.92, "Held-out test set · 30 repeats", ha="center", color=SLATE)
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    fig.savefig(FIGURES_DIR / "feature_importance.png", dpi=180)
    plt.close(fig)


def write_executive_summary(
    metrics: pd.DataFrame,
    policies: pd.DataFrame,
    fairness: pd.DataFrame,
    monitoring: pd.DataFrame,
    champion: str,
    summary: dict[str, object],
) -> None:
    test_metric = metrics[(metrics.model == champion) & (metrics.split == "test")].iloc[0]
    test_policy = policies[(policies.model == champion) & (policies.split == "test")]
    approve_all = test_policy[test_policy.policy == "Approve all"].iloc[0]
    expected_policy = test_policy[test_policy.policy == "Expected-value policy"].iloc[0]
    uplift = (
        expected_policy.value_units_per_100_applications
        / approve_all.value_units_per_100_applications
        - 1
    )
    age = fairness[fairness.audit_dimension == "age_group"].set_index("group")
    selection_gap = age.loc["under_25", "selection_rate"] - age.loc["25_plus", "selection_rate"]
    action_drift = int((monitoring.severity == "action").sum())
    value_ci = summary["test_expected_value_policy"]["value_95_ci"]
    text = f"""# Executive Summary

## Decision

Use **{champion}** as the challenger-backed champion and translate its probability of bad credit into an expected-value approval policy. The model was chosen on the validation set; the test set remained untouched until final reporting.

## Held-out evidence

- ROC-AUC: **{test_metric.roc_auc:.3f}** (95% bootstrap CI **[{test_metric.roc_auc_ci_low:.3f}, {test_metric.roc_auc_ci_high:.3f}]**)
- Gini: **{test_metric.gini:.3f}**
- KS: **{test_metric.ks:.3f}**
- Brier score: **{test_metric.brier_score:.3f}**
- Expected-value policy approval rate: **{expected_policy.approval_rate:.1%}** after prevalence weighting
- Expected bad rate among approvals: **{expected_policy.expected_bad_rate:.1%}**
- Illustrative value per 100 applications: **{expected_policy.value_units_per_100_applications:,.0f}** (95% bootstrap interval **[{value_ci[0]:,.0f}, {value_ci[1]:,.0f}]**)
- Value uplift versus approving all: **{uplift:.0%}** under the stated scenario

## Governance findings

- `age`, `foreign_worker`, and the combined `personal_status_sex` field are excluded from model inputs.
- The raw test-set selection-rate gap for applicants under 25 versus 25+ is **{selection_gap:+.1%}**. This is a diagnostic, not proof of discrimination; the under-25 group has only {int(age.loc['under_25', 'n'])} test rows.
- The foreign-worker subgroup is too small for a reliable test-set comparison and is flagged automatically.
- The synthetic monitoring demonstration produces **{action_drift}** feature(s) at the action-level PSI threshold. It validates the monitoring code, not a claim about live production drift.

## Boundary of the evidence

South German Credit is an accepted-only, stratified sample from 1973–1975. Bad credits were deliberately oversampled to 30%, the amount field is a transformed historical proxy, sex cannot be reconstructed reliably, and no application date exists. Absolute PD, monetary return, fairness, reject inference, and temporal stability therefore require contemporary production data before any lending use.
"""
    (REPORTS_DIR / "executive_summary.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()
    result = evaluate(n_bootstrap=args.bootstrap)
    print(json.dumps(result, indent=2))
