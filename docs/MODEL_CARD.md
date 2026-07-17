# Model Card — Beyond the Score

## Intended use

Educational portfolio case demonstrating probability-of-default modeling, calibration, credit policy economics, explanation, subgroup auditing, and monitoring design.

## Prohibited use

This artifact must not be used to approve, decline, price, or collect real consumer credit. It is not validated for any current population, geography, product, legal framework, or protected class.

## Champion

The champion is selected by expected-value policy performance on the validation partition. The currently generated release selects calibrated gradient boosting over logistic regression.

## Inputs

Seventeen application-time fields are modeled. `age`, `foreign_worker`, and `personal_status_sex` are audit-only and programmatically excluded.

## Outputs

- Sample bad-credit probability
- Scenario-adjusted bad-credit probability
- Expected-value estimate in illustrative units
- Approve/decline demonstration decision
- Non-causal global importance and local reason codes

## Validation

- Untouched 20% test partition
- ROC-AUC bootstrap interval
- Gini, KS, average precision, Brier score, log loss, and calibration
- Policy comparison against approve-all and fixed-PD baselines
- Age and foreign-worker subgroup diagnostics with small-sample warnings
- Automated data-contract, metric-direction, policy, drift, and model smoke tests

## Material limitations

- Accepted-only sample creates selection bias and prevents reject inference.
- 1970s German contracts are not representative of modern retail lending.
- Bad outcomes are oversampled; absolute probability requires a production prior.
- No dates are available for out-of-time validation.
- Exposure amounts are transformed and not suitable for monetary forecasting.
- Sex cannot be recovered from the combined source code.
- The foreign-worker subgroup is too small for reliable comparative conclusions.
