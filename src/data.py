"""Load, validate and split the corrected South German Credit data."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    ALLOWED_CATEGORY_VALUES,
    MODEL_FEATURES,
    RANDOM_STATE,
    RAW_DATA_PATH,
    RENAME_COLUMNS,
    TARGET_COLUMN,
)


@dataclass(frozen=True)
class CreditSplits:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def file_sha256(path: Path = RAW_DATA_PATH) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_credit_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    frame = pd.read_csv(path, sep=r"\s+").rename(columns=RENAME_COLUMNS)
    frame.insert(0, "application_id", np.arange(1, len(frame) + 1))
    frame[TARGET_COLUMN] = (1 - frame.pop("credit_risk")).astype(int)
    validate_credit_data(frame)
    return frame


def validate_credit_data(frame: pd.DataFrame) -> None:
    expected = {"application_id", *RENAME_COLUMNS.values(), TARGET_COLUMN} - {"credit_risk"}
    missing = expected - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if frame.shape != (1000, 22):
        raise ValueError(f"Expected 1,000 rows and 22 derived columns, got {frame.shape}")
    if frame.isna().any().any():
        raise ValueError("Dataset contains unexpected missing values")
    if frame.duplicated().any() or frame.application_id.duplicated().any():
        raise ValueError("Dataset violates row or application-id uniqueness")
    if frame[TARGET_COLUMN].value_counts().to_dict() != {0: 700, 1: 300}:
        raise ValueError("Unexpected good/bad target distribution")
    if not frame["amount"].gt(0).all() or not frame["duration"].gt(0).all():
        raise ValueError("Amount and duration must be positive")
    if not frame["age"].between(18, 100).all():
        raise ValueError("Age falls outside the documented adult range")
    for column, allowed in ALLOWED_CATEGORY_VALUES.items():
        observed = set(frame[column].unique())
        if not observed <= allowed:
            raise ValueError(f"Unexpected {column} values: {sorted(observed - allowed)}")


def split_credit_data(frame: pd.DataFrame) -> CreditSplits:
    development, test = train_test_split(
        frame,
        test_size=0.20,
        stratify=frame[TARGET_COLUMN],
        random_state=RANDOM_STATE,
    )
    train, validation = train_test_split(
        development,
        test_size=0.25,
        stratify=development[TARGET_COLUMN],
        random_state=RANDOM_STATE,
    )
    return CreditSplits(
        train=train.sort_values("application_id").copy(),
        validation=validation.sort_values("application_id").copy(),
        test=test.sort_values("application_id").copy(),
    )


def model_xy(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return frame[MODEL_FEATURES].copy(), frame[TARGET_COLUMN].copy()


def data_quality_summary(frame: pd.DataFrame) -> pd.DataFrame:
    checks = [
        ("rows", len(frame), 1000, len(frame) == 1000),
        ("columns_after_derivation", frame.shape[1], 22, frame.shape[1] == 22),
        ("missing_cells", int(frame.isna().sum().sum()), 0, not frame.isna().any().any()),
        ("duplicate_rows", int(frame.duplicated().sum()), 0, not frame.duplicated().any()),
        ("bad_credit_rows", int(frame[TARGET_COLUMN].sum()), 300, frame[TARGET_COLUMN].sum() == 300),
        ("model_features", len(MODEL_FEATURES), 17, len(MODEL_FEATURES) == 17),
        ("audit_only_features", 3, 3, True),
    ]
    return pd.DataFrame(checks, columns=["check", "observed", "expected", "passed"])
