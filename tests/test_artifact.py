from __future__ import annotations

import numpy as np
import pytest

from lead_scoring.artifacts import ModelBundle, load_model_bundle, save_model_bundle
from lead_scoring.data.preparation import prepare_data, select_features
from lead_scoring.modeling import candidates
from lead_scoring.schema import MODEL_FEATURES, TARGET


@pytest.mark.parametrize(
    "model_name", ["logistic_regression", "catboost_depth_4", "xgboost_depth_4"]
)
def test_artifact_loading_and_prediction_schema(raw_frame, tmp_path, model_name):
    frame = prepare_data(raw_frame).frame
    pipeline = next(x.pipeline for x in candidates(42) if x.name == model_name)
    pipeline.fit(select_features(frame), frame[TARGET])
    bundle = ModelBundle(
        pipeline=pipeline,
        model_version="test-v1",
        model_features=tuple(MODEL_FEATURES),
        source_hash="fixture",
        operating_threshold=0.2,
        top_fraction=0.1,
    )
    save_model_bundle(tmp_path, bundle)
    loaded = load_model_bundle(tmp_path)
    probabilities = loaded.pipeline.predict_proba(select_features(frame))[:, 1]
    np.testing.assert_allclose(probabilities, pipeline.predict_proba(select_features(frame))[:, 1])
    assert probabilities.shape == (len(frame),)
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
