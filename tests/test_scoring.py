from __future__ import annotations

import pandas as pd

from lead_scoring.artifacts import ModelBundle
from lead_scoring.config import Settings
from lead_scoring.data.preparation import prepare_data, select_features
from lead_scoring.modeling import candidates
from lead_scoring.schema import MODEL_FEATURES, TARGET
from lead_scoring.scoring import rank_leads, score


def test_priority_ranking_is_stable_and_capacity_based():
    result = rank_leads(
        pd.Series(["L3", "L1", "L2", "L4"]),
        pd.Series([0.8, 0.8, 0.9, 0.1]),
        capacity_fraction=0.5,
    )
    assert result["lead_id"].tolist() == ["L2", "L1", "L3", "L4"]
    assert result["priority_rank"].tolist() == [1, 2, 3, 4]
    assert result["priority_tier"].tolist() == ["call", "call", "backlog", "backlog"]


def test_scoring_does_not_depend_on_target_values(raw_frame, monkeypatch):
    class DatabaseStub:
        written = None

        def write_scores(self, scores, batch):
            self.written = (scores, batch)

    prepared = prepare_data(raw_frame).frame
    pipeline = next(
        candidate.pipeline
        for candidate in candidates(42)
        if candidate.name == "logistic_regression"
    )
    pipeline.fit(select_features(prepared), prepared[TARGET])
    bundle = ModelBundle(
        pipeline=pipeline,
        model_version="test-v1",
        model_features=tuple(MODEL_FEATURES),
        source_hash="training-fixture",
        operating_threshold=0.2,
        top_fraction=0.1,
    )
    monkeypatch.setattr("lead_scoring.scoring.load_model_bundle", lambda _: bundle)
    database = DatabaseStub()

    result = score(
        raw_frame.drop(columns=[TARGET]),
        "scoring-fixture",
        Settings(),
        database,
    )

    assert len(result) == len(raw_frame)
    assert database.written is not None
    assert database.written[1].source_hash == "scoring-fixture"
