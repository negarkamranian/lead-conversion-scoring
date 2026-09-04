from __future__ import annotations

import pandas as pd
import pytest

from lead_scoring.data.preparation import DataValidationError, prepare_data, select_features
from lead_scoring.data.splitting import chronological_split
from lead_scoring.schema import ID_COLUMN, MODEL_FEATURES, TARGET


def test_target_and_identifier_cannot_enter_features(raw_frame):
    prepared = prepare_data(raw_frame).frame
    features = select_features(prepared)
    assert list(features.columns) == MODEL_FEATURES
    assert TARGET not in features
    assert ID_COLUMN not in features


def test_validation_rejects_invalid_binary(raw_frame):
    raw_frame.loc[0, TARGET] = "yes"
    with pytest.raises(ValueError):
        prepare_data(raw_frame)


def test_validation_rejects_invalid_range(raw_frame):
    raw_frame.loc[0, "Discount Percent"] = "101"
    with pytest.raises(DataValidationError, match="Discount Percent"):
        prepare_data(raw_frame)


def test_validation_reports_missing_columns_at_the_data_boundary(raw_frame):
    raw_frame = raw_frame.drop(columns=["Price"])
    with pytest.raises(KeyError, match="Price"):
        prepare_data(raw_frame)


def test_validation_rejects_malformed_numeric_values(raw_frame):
    raw_frame.loc[0, "Price"] = "not-a-number"
    with pytest.raises(ValueError):
        prepare_data(raw_frame)


def test_feature_only_data_does_not_require_the_target(raw_frame):
    feature_only = raw_frame.drop(columns=[TARGET])
    prepared = prepare_data(feature_only, require_target=False)
    assert TARGET not in prepared.frame
    assert list(select_features(prepared.frame).columns) == MODEL_FEATURES


def test_business_deduplication_keeps_earliest(raw_frame):
    duplicate = raw_frame.iloc[[0]].copy()
    duplicate["Created At"] = "2026-01-03 23:59:00"
    prepared = prepare_data(pd.concat([raw_frame, duplicate], ignore_index=True))
    assert prepared.duplicate_ids_removed == 1
    assert len(prepared.frame) == len(raw_frame)


def test_chronological_partitions_do_not_overlap(raw_frame):
    splits = chronological_split(prepare_data(raw_frame).frame)
    ids = [set(part[ID_COLUMN]) for part in (splits.train, splits.validation, splits.test)]
    assert not ids[0] & ids[1]
    assert not ids[0] & ids[2]
    assert not ids[1] & ids[2]
    assert splits.train["Created At"].max() < splits.validation["Created At"].min()
