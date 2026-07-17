# Executive Summary

## Decision

Use **Calibrated Gradient Boosting** as the challenger-backed champion and translate its probability of bad credit into an expected-value approval policy. The model was chosen on the validation set; the test set remained untouched until final reporting.

## Held-out evidence

- ROC-AUC: **0.764** (95% bootstrap CI **[0.689, 0.834]**)
- Gini: **0.527**
- KS: **0.386**
- Brier score: **0.171**
- Expected-value policy approval rate: **64.8%** after prevalence weighting
- Expected bad rate among approvals: **8.1%**
- Illustrative value per 100 applications: **17,179** (95% bootstrap interval **[8,127, 25,507]**)
- Value uplift versus approving all: **226%** under the stated scenario

## Governance findings

- `age`, `foreign_worker`, and the combined `personal_status_sex` field are excluded from model inputs.
- The raw test-set selection-rate gap for applicants under 25 versus 25+ is **-13.2%**. This is a diagnostic, not proof of discrimination; the under-25 group has only 35 test rows.
- The foreign-worker subgroup is too small for a reliable test-set comparison and is flagged automatically.
- The synthetic monitoring demonstration produces **3** feature(s) at the action-level PSI threshold. It validates the monitoring code, not a claim about live production drift.

## Boundary of the evidence

South German Credit is an accepted-only, stratified sample from 1973–1975. Bad credits were deliberately oversampled to 30%, the amount field is a transformed historical proxy, sex cannot be reconstructed reliably, and no application date exists. Absolute PD, monetary return, fairness, reject inference, and temporal stability therefore require contemporary production data before any lending use.
