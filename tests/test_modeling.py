from __future__ import annotations

import numpy as np
import pytest

from lead_scoring.data.preparation import prepare_data, select_features
from lead_scoring.modeling import candidates
from lead_scoring.schema import TARGET


@pytest.mark.parametrize(
    "model_name",
    [
        "logistic_regression",
        "spline_logistic_regression",
        "extra_trees",
        "catboost_depth_4",
        "xgboost_depth_4",
    ],
)
def test_preprocessing_handles_missing_and_unseen_categories(raw_frame, model_name):
    frame = prepare_data(raw_frame).frame
    train = frame.iloc[:45]
    held_out = frame.iloc[45:].copy()
    held_out.loc[held_out.index[0], "Channel"] = "brand_new_channel"
    held_out.loc[held_out.index[1], "City"] = np.nan
    held_out.loc[held_out.index[2], "Price"] = np.nan
    pipeline = next(x.pipeline for x in candidates(42) if x.name == model_name)
    pipeline.fit(select_features(train), train[TARGET])
    probabilities = pipeline.predict_proba(select_features(held_out))[:, 1]
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
