# Methodology

## Decision frame

The model estimates bad-credit probability. A policy then decides whether the expected contribution of approval is positive. Model performance and business policy are evaluated separately.

## Data partition

- Training: 600 contracts
- Validation: 200 contracts
- Test: 200 contracts
- All partitions preserve the source 30% bad-credit share.
- The random split is a benchmark compromise because the source has no application date.
- Model choice uses validation expected value only; test results are reported once.

## Models

1. Logistic regression with standardized numeric fields and one-hot categorical fields.
2. Gradient boosting with the same inputs, followed by five-fold sigmoid probability calibration on the training partition.

The shared feature set prevents the challenger from receiving a feature-budget advantage.

## Sample-to-population prior correction

The source deliberately oversamples bad credits. Policy scenarios therefore adjust calibrated sample odds from source prevalence $\pi_s=0.30$ to assumed operating prevalence $\pi_t=0.15$:

$$
\operatorname{odds}(p_t)
=\operatorname{odds}(p_s)
\times
\frac{\pi_t/(1-\pi_t)}{\pi_s/(1-\pi_s)}
$$

The 15% operating rate is a scenario parameter, not an estimate from this dataset.

## Expected-value policy

For applicant $i$, with adjusted bad probability $p_i$ and amount proxy $A_i$:

$$
EV_i=(1-p_i)mA_i-p_i\,LGD\,A_i-C_{op}
$$

The default scenario uses $m=18\%$, $LGD=65\%$, and $C_{op}=50$ value units. Approve when $EV_i>0$. Because the source amount is a transformed historical proxy, outputs are **illustrative value units**, not monetary forecasts.

## Metrics

- Discrimination: ROC-AUC, Gini, average precision, and KS.
- Probability accuracy: Brier score, log loss, and quintile calibration.
- Policy: prevalence-weighted approval rate, expected bad rate, and value per 100 applications.
- Uncertainty: 1,000 bootstrap resamples for test ROC-AUC and champion policy value.

## Explainability

Global importance is the mean held-out ROC-AUC decrease after permuting one raw feature. Local reason codes replace one feature at a time with its training reference value and measure the score change. Neither method establishes causality.

## Fairness boundary

`age`, `foreign_worker`, and `personal_status_sex` are excluded from modeling. The first two support limited outcome audits. The combined personal-status/sex code cannot recover sex reliably, and small subgroup sizes prevent strong conclusions. Exclusion alone does not guarantee fair outcomes.
