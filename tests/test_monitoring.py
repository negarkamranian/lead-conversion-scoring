import numpy as np
import pandas as pd
import pytest

from lead_scoring.artifacts import METADATA_PATH
from lead_scoring.data.preparation import prepare_data
from lead_scoring.monitoring import build_monitoring_baseline, monitor
from lead_scoring.serialization import read_json, write_json


@pytest.mark.parametrize("with_scores", [True, False])
def test_monitoring_unchanged_data_round_trip(raw_frame, tmp_path, with_scores):
    probabilities = np.linspace(0.05, 0.95, len(raw_frame))
    baseline = build_monitoring_baseline(prepare_data(raw_frame).frame, probabilities)
    write_json(tmp_path / METADATA_PATH, {"monitoring_baseline": baseline.model_dump()})
    scores = (
        pd.DataFrame({"purchase_probability": probabilities}) if with_scores else pd.DataFrame()
    )

    report = monitor(raw_frame, scores, tmp_path)

    assert all(feature.psi == pytest.approx(0) for feature in report.numeric_features.values())
    payload = read_json(tmp_path / "monitoring_report.json")
    assert payload == report.model_dump(mode="json")
    if with_scores:
        assert payload["predictions"]["rows"] == len(raw_frame)
        assert payload["predictions"]["psi"] == pytest.approx(0)
    else:
        assert payload["predictions"] == {}


def test_monitoring_measures_changed_distributions(raw_frame, tmp_path):
    probabilities = np.linspace(0.05, 0.95, len(raw_frame))
    baseline = build_monitoring_baseline(prepare_data(raw_frame).frame, probabilities)
    write_json(tmp_path / METADATA_PATH, {"monitoring_baseline": baseline.model_dump()})
    changed = raw_frame.copy()
    changed["City"] = "new city"
    changed["Price"] = "9000000"

    report = monitor(changed, pd.DataFrame(), tmp_path)

    assert report.categorical_features["City"].psi > 0
    assert report.numeric_features["Price"].psi > 0
