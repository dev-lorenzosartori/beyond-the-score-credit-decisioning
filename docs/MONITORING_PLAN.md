# Monitoring Plan

Monitoring is divided into immediate input checks and delayed outcome checks. Thresholds are operating defaults to be replaced with portfolio-specific control limits.

| Layer | Metric | Default trigger | Response |
|---|---|---|---|
| Data contract | Required columns, type/range, duplicates | any failure | stop scoring and investigate ingestion |
| Missingness | Missing rate by feature | material change from training baseline | inspect upstream source and fallback behavior |
| Population drift | Feature PSI | `<0.10` stable; `0.10–0.25` watch; `≥0.25` action | segment drift, inspect causes, assess retraining |
| Score drift | Score PSI | same thresholds | check calibration and policy impact |
| Decision | Approval-rate change | portfolio control limit | review policy, mix, capacity, and manual overrides |
| Risk | Expected bad rate among approvals | risk-appetite limit | tighten/retune only with approved governance |
| Realized outcome | Bad rate by booking cohort | after maturity window | compare with forecast and vintage expectations |
| Calibration | Brier score and reliability curve | deterioration vs reference | recalibrate or redevelop |
| Ranking | ROC-AUC, Gini, KS | deterioration with confidence interval | investigate features, drift, and labels |
| Fairness | Selection and good-applicant approval gaps | reviewed governance threshold | root-cause and legal/compliance review |

`src/monitoring.py` creates a deterministic drift demonstration by changing amount, duration, and account-status mix. Its output proves that monitoring code fires; it is not production evidence.

Production implementation also needs model/version IDs, decision timestamps, override reasons, adverse-action reason governance, immutable score snapshots, label-availability flags, and retraining approvals.
