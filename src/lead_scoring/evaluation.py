from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from pydantic import BaseModel

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
)

from lead_scoring.metrics import classification_metrics
from lead_scoring.schema import TARGET


class SegmentMetrics(BaseModel):
    """Classification metrics for one categorical segment."""

    segment: str
    rows: int
    prevalence: float
    average_precision: float
    roc_auc: float
    brier_score: float


def segment_metrics(
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    columns: tuple[str, ...] = ("Product Type", "Channel"),
) -> dict[str, list[SegmentMetrics]]:
    result: dict[str, list[SegmentMetrics]] = {}
    scored = frame.copy()
    scored["_score"] = probabilities
    for column in columns:
        groups: list[SegmentMetrics] = []
        for value, group in scored.groupby(column, dropna=False):
            if len(group) < 200 or group[TARGET].nunique() < 2:
                continue
            metrics = classification_metrics(
                group[TARGET].to_numpy(), group["_score"].to_numpy(), threshold=0.5
            )
            groups.append(
                SegmentMetrics(
                    segment=str(value),
                    rows=len(group),
                    prevalence=metrics.prevalence,
                    average_precision=metrics.average_precision,
                    roc_auc=metrics.roc_auc,
                    brier_score=metrics.brier_score,
                )
            )
        result[column] = groups
    return result


def generate_evaluation_charts(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    chart_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    PrecisionRecallDisplay.from_predictions(y_true, probabilities, ax=ax)
    ax.axhline(np.mean(y_true), color="grey", linestyle="--", label="Prevalence")
    ax.legend()
    ax.set_title("Final model: precision-recall")
    fig.tight_layout()
    fig.savefig(chart_dir / "model_precision_recall.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y_true, probabilities, ax=ax)
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--")
    ax.set_title("Final model: ROC")
    fig.tight_layout()
    fig.savefig(chart_dir / "model_roc.png", dpi=140)
    plt.close(fig)

    observed, predicted = calibration_curve(y_true, probabilities, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(predicted, observed, marker="o", label="Model")
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--", label="Ideal")
    ax.set(xlabel="Mean predicted probability", ylabel="Observed frequency", title="Calibration")
    ax.legend()
    fig.tight_layout()
    fig.savefig(chart_dir / "model_calibration.png", dpi=140)
    plt.close(fig)

    order = np.argsort(-probabilities, kind="stable")
    cumulative = np.cumsum(np.asarray(y_true)[order]) / max(1, np.sum(y_true))
    population = np.arange(1, len(y_true) + 1) / len(y_true)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(population, cumulative, label="Model")
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--", label="Random")
    ax.set(
        xlabel="Share of leads contacted",
        ylabel="Share of purchases captured",
        title="Cumulative gains",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(chart_dir / "model_cumulative_gains.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for target, color in [(0, "#7f7f7f"), (1, "#d62728")]:
        ax.hist(
            probabilities[np.asarray(y_true) == target],
            bins=35,
            alpha=0.55,
            density=True,
            color=color,
            label=str(target),
        )
    ax.axvline(threshold, color="black", linestyle="--", label="10% capacity threshold")
    ax.set(
        xlabel="Predicted purchase probability",
        ylabel="Density",
        title="Score distribution by outcome",
    )
    ax.legend(title="Purchased")
    fig.tight_layout()
    fig.savefig(chart_dir / "model_score_distribution.png", dpi=140)
    plt.close(fig)

    predictions = (probabilities >= threshold).astype(int)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay.from_predictions(y_true, predictions, ax=ax, colorbar=False)
    ax.set_title("Confusion matrix at validation capacity threshold")
    fig.tight_layout()
    fig.savefig(chart_dir / "model_confusion_matrix.png", dpi=140)
    plt.close(fig)
