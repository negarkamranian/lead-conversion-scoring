import json
import logging
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd

from lead_scoring.artifacts import (
    METADATA_PATH,
    CategoricalBaseline,
    MonitoringBaseline,
    NumericBaseline,
)
from lead_scoring.data.preparation import prepare_data
from lead_scoring.schema import CATEGORICAL_FEATURES, NUMERIC_FEATURES

logger = logging.getLogger(__name__)

PSI_WARNING = 0.20
MISSING_WARNING = 0.05
UNSEEN_WARNING = 0.01
OUTSIDE_RANGE_WARNING = 0.01
EPSILON = 1e-6
MISSING_CATEGORY = "__MISSING__"


def _histogram(values: npt.ArrayLike) -> tuple[list[float], list[int]]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        raise ValueError("Cannot build a monitoring baseline without finite values")

    edges = np.unique(np.quantile(finite, np.linspace(0, 1, 11)))
    if len(edges) == 1:
        edges = np.array([edges[0], edges[0] + 1.0])

    counts, _ = np.histogram(finite, bins=edges)
    return [float(value) for value in edges], [int(value) for value in counts]


def _categorical_values(values: pd.Series) -> pd.Series:
    return values.fillna(MISSING_CATEGORY).astype(str)


def _categorical_distribution(values: pd.Series) -> dict[str, float]:
    frequencies = _categorical_values(values).value_counts(normalize=True).to_dict()
    return {str(key): float(value) for key, value in frequencies.items()}


def build_monitoring_baseline(
    frame: pd.DataFrame,
    probabilities: npt.ArrayLike,
) -> MonitoringBaseline:
    """Capture training distributions used for later drift comparisons."""
    numeric: dict[str, NumericBaseline] = {}
    for column in NUMERIC_FEATURES:
        values = frame[column].to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        edges, counts = _histogram(values)
        numeric[column] = NumericBaseline(
            missing_rate=float(frame[column].isna().mean()),
            min=float(finite.min()),
            max=float(finite.max()),
            bin_edges=edges,
            bin_counts=counts,
        )

    categorical: dict[str, CategoricalBaseline] = {}
    for column in CATEGORICAL_FEATURES:
        categorical[column] = CategoricalBaseline(
            missing_rate=float(frame[column].isna().mean()),
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


def psi(reference_counts, values, edges):
    if len(edges) < 2:
        return 0

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

    return np.sum((current - reference) * np.log(current / reference))


def categorical_psi(reference, values):
    current = _categorical_distribution(values)

    total = 0

    for key in set(reference) | set(current):
        expected = max(reference.get(key, 0), EPSILON)
        actual = max(current.get(key, 0), EPSILON)

        total += (actual - expected) * np.log(actual / expected)

    return total


def build_monitoring_report(raw, scores, metadata):
    frame = prepare_data(raw).frame

    baseline = MonitoringBaseline.from_metadata(
        metadata,
        NUMERIC_FEATURES,
        CATEGORICAL_FEATURES,
    )

    warnings = []
    numeric_report = {}

    for column in NUMERIC_FEATURES:
        spec = baseline.numeric[column]
        values = frame[column].to_numpy(dtype=float)

        feature_psi = psi(
            np.array(spec.bin_counts),
            values,
            spec.bin_edges,
        )

        missing_rate = frame[column].isna().mean()
        missing_delta = missing_rate - spec.missing_rate

        finite = values[np.isfinite(values)]
        outside_rate = np.mean((finite < spec.min) | (finite > spec.max)) if len(finite) else 0

        numeric_report[column] = {
            "psi": feature_psi,
            "missing_rate": missing_rate,
            "outside_range_rate": outside_rate,
        }

        if feature_psi > PSI_WARNING:
            warnings.append(f"{column}: high PSI")

        if abs(missing_delta) > MISSING_WARNING:
            warnings.append(f"{column}: missing rate changed")

        if outside_rate > OUTSIDE_RANGE_WARNING:
            warnings.append(f"{column}: values outside training range")

    categorical_report = {}

    for column in CATEGORICAL_FEATURES:
        spec = baseline.categorical[column]
        values = _categorical_values(frame[column])

        feature_psi = categorical_psi(
            spec.frequencies,
            frame[column],
        )

        unseen_rate = (~values.isin(spec.frequencies)).mean()
        missing_delta = frame[column].isna().mean() - spec.missing_rate

        categorical_report[column] = {
            "psi": feature_psi,
            "unseen_rate": unseen_rate,
        }

        if feature_psi > PSI_WARNING:
            warnings.append(f"{column}: high PSI")

        if abs(missing_delta) > MISSING_WARNING:
            warnings.append(f"{column}: missing rate changed")

        if unseen_rate > UNSEEN_WARNING:
            warnings.append(f"{column}: unseen categories")

    prediction_report = {}

    if not scores.empty:
        probabilities = scores["purchase_probability"].to_numpy(dtype=float)

        score_psi = psi(
            np.array(baseline.prediction_bin_counts),
            probabilities,
            baseline.prediction_bin_edges,
        )

        prediction_report = {
            "rows": len(scores),
            "mean": probabilities.mean(),
            "std": probabilities.std(),
            "psi": score_psi,
        }

        if score_psi > PSI_WARNING:
            warnings.append("Prediction PSI is high")

    return {
        "status": "warning" if warnings else "ok",
        "numeric_features": numeric_report,
        "categorical_features": categorical_report,
        "predictions": prediction_report,
        "warnings": warnings,
    }


def monitor(raw, scores, artifact_dir: Path):
    with open(artifact_dir / METADATA_PATH) as file:
        metadata = json.load(file)

    report = build_monitoring_report(raw, scores, metadata)

    with open(artifact_dir / "monitoring_report.json", "w") as file:
        json.dump(report, file, indent=2, default=float)

    logger.info("Monitoring status=%s", report["status"])

    return report
