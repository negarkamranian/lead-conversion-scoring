from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from lead_scoring.schema import (
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    ID_COLUMN,
    MODEL_FEATURES,
    NUMERIC_FEATURES,
    RANGES,
    TARGET,
    TIME_COLUMN,
)

RAW_NUMERIC = [c for c in NUMERIC_FEATURES if not c.startswith("Created ")]


class DataValidationError(ValueError):
    """Raised when source data violates the modeling contract."""


@dataclass(frozen=True)
class PreparedData:
    frame: pd.DataFrame
    duplicate_ids_removed: int


def _coerce_types(raw: pd.DataFrame, *, require_target: bool) -> pd.DataFrame:
    """Parse external tabular values and let pandas reject malformed values."""
    frame = raw.copy()
    numeric_columns = RAW_NUMERIC + ([TARGET] if require_target else [])
    for column in numeric_columns:
        source = frame[column].replace(r"^\s*$", pd.NA, regex=True)
        frame[column] = pd.to_numeric(source, errors="raise")
    frame[TIME_COLUMN] = pd.to_datetime(frame[TIME_COLUMN], errors="raise", utc=True)
    for column in CATEGORICAL_FEATURES:
        frame[column] = frame[column].replace("", np.nan).astype(object)
    frame[ID_COLUMN] = frame[ID_COLUMN].replace(r"^\s*$", pd.NA, regex=True).astype("string")
    return frame


def _validate_data(frame: pd.DataFrame, *, require_target: bool) -> None:
    if frame[ID_COLUMN].isna().any():
        raise DataValidationError(f"{ID_COLUMN} contains missing values")
    if frame[TIME_COLUMN].isna().any():
        count = int(frame[TIME_COLUMN].isna().sum())
        raise DataValidationError(f"{TIME_COLUMN} contains {count} invalid timestamps")
    if require_target:
        invalid_target = ~frame[TARGET].isin([0, 1])
        if invalid_target.any():
            values = frame.loc[invalid_target, TARGET].drop_duplicates().tolist()[:5]
            raise DataValidationError(f"{TARGET} must be binary 0/1; invalid values: {values}")
        if frame[TARGET].nunique() != 2:
            raise DataValidationError(f"{TARGET} must contain both classes")
    for column in BINARY_FEATURES:
        invalid = frame[column].notna() & ~frame[column].isin([0, 1])
        if invalid.any():
            raise DataValidationError(f"{column} contains values outside 0/1")
    for column, (lower, upper) in RANGES.items():
        invalid = frame[column].notna() & ~frame[column].between(lower, upper)
        if invalid.any():
            sample = frame.loc[invalid, column].head(3).tolist()
            raise DataValidationError(
                f"{column} has {int(invalid.sum())} values outside [{lower}, {upper}]; "
                f"sample={sample}"
            )


def prepare_data(raw: pd.DataFrame, *, require_target: bool = True) -> PreparedData:
    """Validate, deduplicate, and derive deterministic model-time features."""
    frame = _coerce_types(raw, require_target=require_target)
    _validate_data(frame, require_target=require_target)
    before = len(frame)
    frame = frame.sort_values([TIME_COLUMN, ID_COLUMN], kind="mergesort")
    frame = frame.drop_duplicates(subset=[ID_COLUMN], keep="first").copy()
    frame["Created Hour"] = frame[TIME_COLUMN].dt.hour.astype(float)
    frame["Created Day Of Week"] = frame[TIME_COLUMN].dt.dayofweek.astype(float)
    return PreparedData(
        frame=frame.reset_index(drop=True), duplicate_ids_removed=before - len(frame)
    )


def select_features(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[:, MODEL_FEATURES].copy()
