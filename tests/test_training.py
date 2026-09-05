import pytest

from lead_scoring.metrics import ValidationMetrics
from lead_scoring.training import select_model


@pytest.mark.parametrize(
    ("average_precision", "log_loss", "expected"),
    [(0.505, 0.495, "logistic_regression"), (0.6, 0.5, "challenger"), (0.505, 0.4, "challenger")],
)
def test_model_selection_preserves_logistic_tie_break(average_precision, log_loss, expected):
    logistic = ValidationMetrics(
        average_precision=0.5,
        roc_auc=0.7,
        log_loss=0.5,
        brier_score=0.2,
        precision_at_k=0.8,
        recall_at_k=0.1,
        lift_at_k=2,
    )
    challenger = logistic.model_copy(
        update={"average_precision": average_precision, "log_loss": log_loss}
    )
    dummy = logistic.model_copy(update={"average_precision": 1.0})

    assert (
        select_model(
            {
                "dummy_prior": dummy,
                "logistic_regression": logistic,
                "challenger": challenger,
            }
        )
        == expected
    )
