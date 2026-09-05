import math

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

CAPACITY_LEVELS = (0.01, 0.05, 0.10, 0.20)


class TopKMetrics(BaseModel):
    """Performance when only the highest-scoring fraction is selected."""

    fraction: float
    k: int
    precision_at_k: float
    recall_at_k: float
    lift_at_k: float


class ProbabilityMetrics(BaseModel):
    """Threshold-independent probability-quality metrics."""

    average_precision: float
    roc_auc: float
    log_loss: float
    brier_score: float


class ValidationMetrics(ProbabilityMetrics):
    precision_at_k: float
    recall_at_k: float
    lift_at_k: float


class ClassificationMetrics(ProbabilityMetrics):
    """Overall and capacity-constrained binary classification metrics."""

    rows: int
    prevalence: float
    operating_threshold: float
    precision: float
    recall: float
    f1: float
    confusion_matrix: list[list[int]]
    capacity: dict[str, TopKMetrics]


class BootstrapInterval(BaseModel):
    """Percentile bootstrap confidence intervals for ranking metrics."""

    average_precision_95pct: list[float]
    roc_auc_95pct: list[float]


class EvaluationMetrics(ClassificationMetrics):
    uncertainty: BootstrapInterval


def validate_metric_inputs(
    y: npt.ArrayLike,
    scores: npt.ArrayLike,
) -> tuple[npt.NDArray[np.int_], npt.NDArray[np.float64]]:
    y_array = np.asarray(y)
    scores_array = np.asarray(scores)

    if y_array.ndim != 1 or scores_array.ndim != 1:
        raise ValueError("y and scores must be one-dimensional")
    if len(y_array) != len(scores_array):
        raise ValueError("y and scores must have equal lengths")
    if not len(y_array):
        raise ValueError("y and scores must not be empty")
    if not np.isin(y_array, [0, 1]).all():
        raise ValueError("y must contain only binary labels (0 and 1)")
    if not np.isfinite(scores_array).all() or ((scores_array < 0) | (scores_array > 1)).any():
        raise ValueError("scores must be finite probabilities between 0 and 1")

    return y_array.astype(int), scores_array.astype(float)


def top_k_metrics(
    y: npt.ArrayLike,
    scores: npt.ArrayLike,
    fraction: float,
) -> TopKMetrics:
    y, scores = validate_metric_inputs(y, scores)
    k = math.ceil(len(y) * fraction)

    order = np.argsort(-scores)
    chosen = y[order[:k]]

    precision = chosen.mean()
    prevalence = y.mean()

    return TopKMetrics(
        fraction=fraction,
        k=k,
        precision_at_k=precision,
        recall_at_k=chosen.sum() / y.sum() if y.sum() else 0,
        lift_at_k=precision / prevalence if prevalence else 0,
    )


def _probability_metrics(
    y: npt.NDArray[np.int_],
    scores: npt.NDArray[np.float64],
) -> ProbabilityMetrics:
    if len(np.unique(y)) < 2:
        raise ValueError("y must contain both classes to calculate probability metrics")
    return ProbabilityMetrics(
        average_precision=average_precision_score(y, scores),
        roc_auc=roc_auc_score(y, scores),
        log_loss=log_loss(y, scores),
        brier_score=brier_score_loss(y, scores),
    )


def probability_metrics(y: npt.ArrayLike, scores: npt.ArrayLike) -> ProbabilityMetrics:
    """Calculate the shared probability metrics used for selection and evaluation."""
    y_array, scores_array = validate_metric_inputs(y, scores)
    return _probability_metrics(y_array, scores_array)


def classification_metrics(
    y: npt.ArrayLike,
    scores: npt.ArrayLike,
    threshold: float,
) -> ClassificationMetrics:
    y, scores = validate_metric_inputs(y, scores)
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be in the interval [0, 1]")

    predictions = (scores >= threshold).astype(int)
    probability = _probability_metrics(y, scores)

    return ClassificationMetrics(
        **probability.model_dump(),
        rows=len(y),
        prevalence=y.mean(),
        operating_threshold=threshold,
        precision=precision_score(y, predictions, zero_division=0),
        recall=recall_score(y, predictions, zero_division=0),
        f1=f1_score(y, predictions, zero_division=0),
        confusion_matrix=confusion_matrix(y, predictions).tolist(),
        capacity={f"top_{int(f * 100)}pct": top_k_metrics(y, scores, f) for f in CAPACITY_LEVELS},
    )


def bootstrap_interval(
    y: npt.ArrayLike,
    scores: npt.ArrayLike,
    seed: int,
    samples: int = 200,
) -> BootstrapInterval:
    y_array, scores_array = validate_metric_inputs(y, scores)
    if samples <= 0:
        raise ValueError("samples must be positive")

    rng = np.random.default_rng(seed)

    ap: list[float] = []
    auc: list[float] = []

    for _ in range(samples):
        indices = rng.integers(0, len(y_array), len(y_array))

        if len(np.unique(y_array[indices])) < 2:
            continue

        ap.append(float(average_precision_score(y_array[indices], scores_array[indices])))
        auc.append(float(roc_auc_score(y_array[indices], scores_array[indices])))

    if not ap:
        raise ValueError("bootstrap samples must contain both classes")

    return BootstrapInterval(
        average_precision_95pct=[float(value) for value in np.quantile(ap, [0.025, 0.975])],
        roc_auc_95pct=[float(value) for value in np.quantile(auc, [0.025, 0.975])],
    )
