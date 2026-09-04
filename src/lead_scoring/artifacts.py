from pathlib import Path

import joblib
from pydantic import BaseModel, model_validator
from sklearn.pipeline import Pipeline

from lead_scoring.schema import MODEL_FEATURES

MODEL_PATH = "model.joblib"
METADATA_PATH = "model_metadata.json"


class NumericBaseline(BaseModel):
    missing_rate: float
    min: float
    max: float
    bin_edges: list[float]
    bin_counts: list[int]


class CategoricalBaseline(BaseModel):
    missing_rate: float
    frequencies: dict[str, float]


class MonitoringBaseline(BaseModel):
    rows: int
    numeric: dict[str, NumericBaseline]
    categorical: dict[str, CategoricalBaseline]
    prediction_bin_edges: list[float]
    prediction_bin_counts: list[int]

    @classmethod
    def from_metadata(cls, metadata, numeric_columns, categorical_columns):
        if "monitoring_baseline" not in metadata:
            raise ValueError(
                "Model metadata has no monitoring baseline; retrain the model before monitoring"
            )
        baseline = cls.model_validate(metadata["monitoring_baseline"])

        missing = (
            set(numeric_columns) - baseline.numeric.keys()
            | set(categorical_columns) - baseline.categorical.keys()
        )

        if missing:
            raise ValueError(f"Monitoring baseline is missing features: {sorted(missing)}")

        return baseline


class ModelBundle(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    pipeline: Pipeline
    model_version: str
    model_features: tuple[str, ...]
    source_hash: str
    operating_threshold: float
    top_fraction: float

    @model_validator(mode="after")
    def validate_features(self):
        if self.model_features != tuple(MODEL_FEATURES):
            raise ValueError("Model feature schema does not match the application")
        return self


def save_model_bundle(artifact_dir: Path, bundle: ModelBundle):
    joblib.dump(bundle, artifact_dir / MODEL_PATH)


def load_model_bundle(artifact_dir: Path):
    return joblib.load(artifact_dir / MODEL_PATH)
