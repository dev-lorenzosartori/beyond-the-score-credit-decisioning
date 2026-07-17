"""Build the reader-facing notebook with nbformat."""

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks/beyond_the_score_credit_decisioning.ipynb"


def build() -> None:
    notebook = nbf.v4.new_notebook()
    notebook["metadata"]["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    notebook["metadata"]["language_info"] = {"name": "python", "version": "3.12"}
    notebook["cells"] = [
        nbf.v4.new_markdown_cell(
            """# Beyond the Score — Credit Decisioning Case

## tl;dr

Calibrated gradient boosting was selected on validation expected value. On the untouched test set it achieved ROC-AUC **0.764**, Gini **0.527**, KS **0.386**, and Brier **0.171**. Under the explicit 15% target bad-rate scenario, the expected-value policy produced **17,179 illustrative value units per 100 applications**, 226% above approve-all. The result is a benchmark demonstration, not a lending policy."""
        ),
        nbf.v4.new_markdown_cell(
            """## Context & Methods

The model estimates bad-credit probability; a separate policy converts probability into a decision. Logistic regression and calibrated gradient boosting share the same 17 application-time inputs. Model selection uses the validation partition only, followed by one held-out test report.

### Key Assumptions

- Source bad-rate prior: 30% by stratified sampling design.
- Illustrative operating prior: 15%.
- Good-account margin: 18% of the transformed amount proxy.
- Loss given default: 65% of the proxy.
- Operating cost: 50 value units.
- No timestamp exists, so the split cannot be out-of-time."""
        ),
        nbf.v4.new_code_cell(
            """from pathlib import Path
import pandas as pd
from IPython.display import Image

from src.data import data_quality_summary, load_credit_data, split_credit_data
from src.train_evaluate import evaluate

ROOT = Path.cwd()
pd.set_option("display.max_columns", 30)"""
        ),
        nbf.v4.new_markdown_cell("## Data\n\nThe source is the corrected UCI South German Credit release. The checks below lock its grain, completeness, target distribution, and modeling contract."),
        nbf.v4.new_code_cell(
            """credit = load_credit_data()
splits = split_credit_data(credit)
display(data_quality_summary(credit))
pd.DataFrame({
    "partition": ["train", "validation", "test"],
    "rows": [len(splits.train), len(splits.validation), len(splits.test)],
    "bad_credits": [splits.train.bad_credit.sum(), splits.validation.bad_credit.sum(), splits.test.bad_credit.sum()],
})"""
        ),
        nbf.v4.new_markdown_cell("## Results\n\nThe evaluation regenerates every report and figure. Bootstrap intervals use 1,000 deterministic resamples."),
        nbf.v4.new_code_cell("summary = evaluate(n_bootstrap=1000)\nsummary"),
        nbf.v4.new_code_cell(
            """metrics = pd.read_csv(ROOT / "reports/model_metrics.csv")
metrics[metrics.split == "test"].round(3)"""
        ),
        nbf.v4.new_code_cell(
            """policies = pd.read_csv(ROOT / "reports/policy_comparison.csv")
policies[(policies.model == summary["champion"]) & (policies.split == "test")].round(3)"""
        ),
        nbf.v4.new_code_cell(
            """display(Image(filename=ROOT / "reports/figures/model_comparison.png"))
display(Image(filename=ROOT / "reports/figures/policy_value_comparison.png"))"""
        ),
        nbf.v4.new_markdown_cell("### Governance diagnostics\n\nProtected/problematic attributes are excluded from model inputs. Group metrics are diagnostics with explicit sample-size warnings, and the monitoring batch is synthetic by design."),
        nbf.v4.new_code_cell(
            """fairness = pd.read_csv(ROOT / "reports/fairness_audit.csv")
display(fairness.round(3))
display(pd.read_csv(ROOT / "reports/monitoring_demo.csv").head(10).round(3))"""
        ),
        nbf.v4.new_code_cell(
            """display(Image(filename=ROOT / "reports/figures/calibration.png"))
display(Image(filename=ROOT / "reports/figures/fairness_age_audit.png"))
display(Image(filename=ROOT / "reports/figures/feature_importance.png"))"""
        ),
        nbf.v4.new_markdown_cell(
            """## Takeaways

1. The nonlinear challenger improved held-out discrimination and Brier score, but uncertainty remains material at n=200.
2. A calibrated risk estimate is not an approval rule. Explicit economics materially changed approval and portfolio-value outcomes.
3. Prior correction is mandatory before applying sample-calibrated probabilities to a different operating prevalence.
4. The observed age-group gap requires investigation but cannot establish discrimination with this small, historical, accepted-only sample.
5. Real deployment requires contemporary application and performance data, reject inference strategy, temporal validation, legal review, governed reason codes, and delayed-outcome monitoring."""
        ),
    ]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, OUTPUT)


if __name__ == "__main__":
    build()
