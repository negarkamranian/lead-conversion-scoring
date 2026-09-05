import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from pydantic import BaseModel

from lead_scoring.artifacts import (
    METADATA_PATH,
    CategoricalBaseline,
    MonitoringBaseline,
    NumericBaseline,
)
from lead_scoring.data.preparation import prepare_data
from lead_scoring.schema import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from lead_scoring.serialization import read_json, write_json

logger = logging.getLogger(__name__)

EPSILON = 1e-6
MISSING_CATEGORY = "__MISSING__"


class FeatureDrift(BaseModel):
    psi: float


class PredictionDrift(BaseModel):
    rows: int
    mean: float
    std: float
    psi: float


class MonitoringReport(BaseModel):
    numeric_features: dict[str, FeatureDrift]
    categorical_features: dict[str, FeatureDrift]
    predictions: PredictionDrift | dict[str, float]


def _histogram(values: npt.ArrayLike) -> tuple[list[float], list[int]]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]

    edges = np.unique(np.quantile(finite, np.linspace(0, 1, 11)))
    if len(edges) == 1:
        edges = np.array([edges[0], edges[0] + 1.0])

    counts, _ = np.histogram(finite, bins=edges)
    return [float(value) for value in edges], [int(value) for value in counts]


def _categorical_distribution(values: pd.Series) -> dict[str, float]:
    frequencies = values.fillna(MISSING_CATEGORY).astype(str).value_counts(normalize=True)
    return {str(key): float(value) for key, value in frequencies.items()}


def build_monitoring_baseline(
    frame: pd.DataFrame,
    probabilities: npt.ArrayLike,
) -> MonitoringBaseline:
    """Capture training distributions used for later drift comparisons."""
    numeric: dict[str, NumericBaseline] = {}
    for column in NUMERIC_FEATURES:
        values = frame[column].to_numpy(dtype=float)
        edges, counts = _histogram(values)
        numeric[column] = NumericBaseline(
            bin_edges=edges,
            bin_counts=counts,
        )

    categorical: dict[str, CategoricalBaseline] = {}
    for column in CATEGORICAL_FEATURES:
        categorical[column] = CategoricalBaseline(
            frequencies=_categorical_distribution(frame[column]),
        )

    prediction_edges, prediction_counts = _histogram(probabilities)
    return MonitoringBaseline(
        rows=len(frame),
        numeric=numeric,
        categorical=categorical,
        prediction_bin_edges=prediction_edges,
        prediction_bin_counts=prediction_counts,
    )


def psi(reference_counts: npt.ArrayLike, values: npt.ArrayLike, edges: npt.ArrayLike) -> float:
    reference_counts = np.asarray(reference_counts, dtype=float)
    values = np.asarray(values, dtype=float)
    bins = np.array(edges, dtype=float)

    bins[0], bins[-1] = -np.inf, np.inf

    current_counts, _ = np.histogram(
        values[np.isfinite(values)],
        bins=bins,
    )

    reference = reference_counts / max(reference_counts.sum(), 1)
    current = current_counts / max(current_counts.sum(), 1)

    reference = np.clip(reference, EPSILON, None)
    current = np.clip(current, EPSILON, None)

    return float(np.sum((current - reference) * np.log(current / reference)))


def categorical_psi(reference: Mapping[str, float], values: pd.Series) -> float:
    current = _categorical_distribution(values)

    total = 0.0

    for key in set(reference) | set(current):
        expected = max(reference.get(key, 0), EPSILON)
        actual = max(current.get(key, 0), EPSILON)

        total += (actual - expected) * np.log(actual / expected)

    return float(total)


def build_monitoring_report(
    raw: pd.DataFrame, scores: pd.DataFrame, metadata: Mapping[str, Any]
) -> MonitoringReport:
    frame = prepare_data(raw).frame

    baseline = MonitoringBaseline.model_validate(metadata["monitoring_baseline"])

    numeric_report = {
        column: FeatureDrift(
            psi=psi(
                baseline.numeric[column].bin_counts,
                frame[column].to_numpy(dtype=float),
                baseline.numeric[column].bin_edges,
            )
        )
        for column in NUMERIC_FEATURES
    }
    categorical_report = {
        column: FeatureDrift(
            psi=categorical_psi(baseline.categorical[column].frequencies, frame[column])
        )
        for column in CATEGORICAL_FEATURES
    }

    prediction_report: PredictionDrift | dict[str, float] = {}

    if not scores.empty:
        probabilities = scores["purchase_probability"].to_numpy(dtype=float)

        score_psi = psi(
            np.array(baseline.prediction_bin_counts),
            probabilities,
            baseline.prediction_bin_edges,
        )

        prediction_report = PredictionDrift(
            rows=len(scores),
            mean=probabilities.mean(),
            std=probabilities.std(),
            psi=score_psi,
        )

    return MonitoringReport(
        numeric_features=numeric_report,
        categorical_features=categorical_report,
        predictions=prediction_report,
    )


def monitor(raw: pd.DataFrame, scores: pd.DataFrame, artifact_dir: Path) -> MonitoringReport:
    metadata = read_json(artifact_dir / METADATA_PATH)

    report = build_monitoring_report(raw, scores, metadata)

    write_json(artifact_dir / "monitoring_report.json", report)

    logger.info("Monitoring report saved")

    return report
