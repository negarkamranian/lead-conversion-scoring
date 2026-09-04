import hashlib
import json
import logging
from datetime import UTC, datetime

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from lead_scoring.artifacts import (
    METADATA_PATH,
    ModelBundle,
    load_model_bundle,
    save_model_bundle,
)
from lead_scoring.data.preparation import prepare_data, select_features
from lead_scoring.data.splitting import chronological_split, split_summary
from lead_scoring.metrics import bootstrap_interval, classification_metrics, probability_metrics
from lead_scoring.modeling import candidates, linear_preprocessor
from lead_scoring.monitoring import build_monitoring_baseline
from lead_scoring.schema import (
    CATEGORICAL_FEATURES,
    MODEL_FEATURES,
    NUMERIC_FEATURES,
    TARGET,
)

logger = logging.getLogger(__name__)

QUESTIONABLE_FEATURES = {
    "Insurance Company",
    "Payment Type",
    "Price",
    "Discount Percent",
    "Incoming Call Last 24h",
    "Expected Margin",
}


def select_model(results):
    non_dummy = {name: metrics for name, metrics in results.items() if name != "dummy_prior"}
    best = max(non_dummy, key=lambda name: non_dummy[name]["average_precision"])

    logistic = non_dummy["logistic_regression"]
    leader = non_dummy[best]

    if (
        leader["average_precision"] - logistic["average_precision"] <= 0.01
        and logistic["log_loss"] - leader["log_loss"] <= 0.01
    ):
        return "logistic_regression"

    return best


def train(raw, source_hash, settings):
    prepared = prepare_data(raw)
    splits = chronological_split(prepared.frame)

    x_train = select_features(splits.train)
    y_train = splits.train[TARGET].to_numpy()
    x_validation = select_features(splits.validation)
    y_validation = splits.validation[TARGET].to_numpy()

    fitted = {}
    comparison = {}
    predictions = {}

    for candidate in candidates(settings.random_seed):
        logger.info("Training %s", candidate.name)

        candidate.pipeline.fit(x_train, y_train)
        probabilities = candidate.pipeline.predict_proba(x_validation)[:, 1]

        fitted[candidate.name] = candidate.pipeline
        predictions[candidate.name] = probabilities
        comparison[candidate.name] = probability_metrics(y_validation, probabilities).model_dump(
            mode="json"
        )

    numeric = [x for x in NUMERIC_FEATURES if x not in QUESTIONABLE_FEATURES]
    categorical = [x for x in CATEGORICAL_FEATURES if x not in QUESTIONABLE_FEATURES]
    features = numeric + categorical

    conservative_model = Pipeline(
        [
            ("preprocess", linear_preprocessor(numeric, categorical)),
            (
                "model",
                LogisticRegression(
                    C=0.5,
                    max_iter=1000,
                    random_state=settings.random_seed,
                ),
            ),
        ]
    )

    conservative_model.fit(x_train[features], y_train)
    conservative_predictions = conservative_model.predict_proba(x_validation[features])[:, 1]

    selected_name = select_model(comparison)
    selected = fitted[selected_name]
    selected_predictions = predictions[selected_name]

    threshold = np.quantile(
        selected_predictions,
        1 - settings.top_fraction,
    )

    config = {
        "features": MODEL_FEATURES,
        "selected_model": selected_name,
        "seed": settings.random_seed,
        "top_fraction": settings.top_fraction,
    }

    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()

    version = f"v1-{source_hash[:8]}-{config_hash[:8]}"
    summary = split_summary(splits)
    summary_payload = summary.model_dump(mode="json")
    training_probabilities = selected.predict_proba(x_train)[:, 1]
    monitoring_baseline = build_monitoring_baseline(splits.train, training_probabilities)

    metadata = {
        "model_version": version,
        "trained_at": datetime.now(UTC).isoformat(),
        "source_hash": source_hash,
        "config_hash": config_hash,
        "selected_model": selected_name,
        "candidate_validation_metrics": comparison,
        "questionable_feature_sensitivity": {
            "excluded_features": sorted(QUESTIONABLE_FEATURES),
            "conservative_logistic_validation_metrics": probability_metrics(
                y_validation, conservative_predictions
            ).model_dump(mode="json"),
            "full_logistic_validation_metrics": comparison["logistic_regression"],
        },
        "operating_threshold": threshold,
        "split_summary": summary_payload,
        "monitoring_baseline": monitoring_baseline.model_dump(mode="json"),
        "test_metrics": None,
    }

    bundle = ModelBundle(
        pipeline=selected,
        model_version=version,
        model_features=tuple(MODEL_FEATURES),
        source_hash=source_hash,
        operating_threshold=threshold,
        top_fraction=settings.top_fraction,
    )

    save_model_bundle(settings.artifact_dir, bundle)

    with open(settings.artifact_dir / METADATA_PATH, "w") as file:
        json.dump(metadata, file, indent=2)

    with open(settings.artifact_dir / "split_summary.json", "w") as file:
        json.dump(summary_payload, file, indent=2)

    with open(settings.artifact_dir / "validation_model_comparison.json", "w") as file:
        json.dump(comparison, file, indent=2)

    logger.info("Selected %s", selected_name)

    return metadata


def evaluate(raw, source_hash, settings):
    bundle = load_model_bundle(settings.artifact_dir)

    if bundle.source_hash != source_hash:
        raise ValueError("Model was trained on a different dataset")

    splits = chronological_split(prepare_data(raw).frame)

    x_test = select_features(splits.test)
    y_test = splits.test[TARGET].to_numpy()
    probabilities = bundle.pipeline.predict_proba(x_test)[:, 1]

    metrics = classification_metrics(
        y_test,
        probabilities,
        bundle.operating_threshold,
    )

    metrics["uncertainty"] = bootstrap_interval(
        y_test,
        probabilities,
        seed=settings.random_seed,
        samples=200,
    )

    from lead_scoring.evaluation import generate_evaluation_charts, segment_metrics

    metrics["segments"] = segment_metrics(splits.test, probabilities)

    generate_evaluation_charts(
        y_test,
        probabilities,
        bundle.operating_threshold,
        settings.chart_dir,
    )
    metrics_payload = metrics.model_dump(mode="json")

    with open(settings.artifact_dir / "test_metrics.json", "w") as file:
        json.dump(metrics_payload, file, indent=2)

    with open(settings.artifact_dir / METADATA_PATH) as file:
        metadata = json.load(file)

    metadata["test_metrics"] = metrics_payload

    with open(settings.artifact_dir / METADATA_PATH, "w") as file:
        json.dump(metadata, file, indent=2)

    return metrics
