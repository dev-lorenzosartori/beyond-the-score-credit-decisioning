"""Project-wide constants and business-scenario assumptions."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = ROOT / "data/raw/SouthGermanCredit.asc"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR = ROOT / "models"

RANDOM_STATE = 42
SOURCE_BAD_RATE = 0.30
TARGET_BAD_RATE = 0.15
GOOD_MARGIN_RATE = 0.18
LOSS_GIVEN_DEFAULT = 0.65
OPERATING_COST = 50.0
FIXED_PD_CUTOFF = 0.15

RENAME_COLUMNS = {
    "laufkont": "status",
    "laufzeit": "duration",
    "moral": "credit_history",
    "verw": "purpose",
    "hoehe": "amount",
    "sparkont": "savings",
    "beszeit": "employment_duration",
    "rate": "installment_rate",
    "famges": "personal_status_sex",
    "buerge": "other_debtors",
    "wohnzeit": "present_residence",
    "verm": "property",
    "alter": "age",
    "weitkred": "other_installment_plans",
    "wohn": "housing",
    "bishkred": "number_credits",
    "beruf": "job",
    "pers": "people_liable",
    "telef": "telephone",
    "gastarb": "foreign_worker",
    "kredit": "credit_risk",
}

TARGET_COLUMN = "bad_credit"
AUDIT_ONLY_COLUMNS = ["age", "foreign_worker", "personal_status_sex"]
NUMERIC_FEATURES = ["duration", "amount"]
CATEGORICAL_FEATURES = [
    "status",
    "credit_history",
    "purpose",
    "savings",
    "employment_duration",
    "installment_rate",
    "other_debtors",
    "present_residence",
    "property",
    "other_installment_plans",
    "housing",
    "number_credits",
    "job",
    "people_liable",
    "telephone",
]
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

ALLOWED_CATEGORY_VALUES = {
    "status": {1, 2, 3, 4},
    "credit_history": {0, 1, 2, 3, 4},
    "purpose": set(range(11)),
    "savings": {1, 2, 3, 4, 5},
    "employment_duration": {1, 2, 3, 4, 5},
    "installment_rate": {1, 2, 3, 4},
    "personal_status_sex": {1, 2, 3, 4},
    "other_debtors": {1, 2, 3},
    "present_residence": {1, 2, 3, 4},
    "property": {1, 2, 3, 4},
    "other_installment_plans": {1, 2, 3},
    "housing": {1, 2, 3},
    "number_credits": {1, 2, 3, 4},
    "job": {1, 2, 3, 4},
    "people_liable": {1, 2},
    "telephone": {1, 2},
    "foreign_worker": {1, 2},
}
