from pathlib import Path

import joblib
from pydantic import BaseModel
from sklearn.pipeline import Pipeline

MODEL_PATH = "model.joblib"
METADATA_PATH = "model_metadata.json"


class NumericBaseline(BaseModel):
    bin_edges: list[float]
    bin_counts: list[int]


class CategoricalBaseline(BaseModel):
    frequencies: dict[str, float]


class MonitoringBaseline(BaseModel):
    rows: int
    numeric: dict[str, NumericBaseline]
    categorical: dict[str, CategoricalBaseline]
    prediction_bin_edges: list[float]
    prediction_bin_counts: list[int]


class ModelBundle(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    pipeline: Pipeline
    model_version: str
    model_features: tuple[str, ...]
    source_hash: str
    operating_threshold: float
    top_fraction: float


def save_model_bundle(artifact_dir: Path, bundle: ModelBundle) -> None:
    joblib.dump(bundle, artifact_dir / MODEL_PATH)


def load_model_bundle(artifact_dir: Path) -> ModelBundle:
    return joblib.load(artifact_dir / MODEL_PATH)
