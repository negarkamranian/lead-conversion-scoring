import numpy as np
import pytest

from lead_scoring.metrics import classification_metrics, top_k_metrics


def test_top_k_metrics_and_lift():
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0])
    result = top_k_metrics(y, scores, 0.2)
    assert result.k == 2
    assert result.precision_at_k == 1.0
    assert result.recall_at_k == 1.0
    assert result.lift_at_k == 5.0


def test_metric_schema_contains_capacity_levels():
    y = np.array([0, 1, 0, 1])
    scores = np.array([0.1, 0.8, 0.2, 0.7])
    metrics = classification_metrics(y, scores, 0.5)
    assert "top_10pct" in metrics.capacity
    assert metrics.confusion_matrix == [[2, 0], [0, 2]]


def test_metrics_reject_misaligned_or_invalid_inputs():
    with pytest.raises(ValueError, match="equal lengths"):
        top_k_metrics(np.array([0, 1]), np.array([0.2]), 0.5)
    with pytest.raises(ValueError, match="threshold"):
        classification_metrics(np.array([0, 1]), np.array([0.2, 0.8]), 1.1)
