from __future__ import annotations

from dataclasses import dataclass

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from lead_scoring.schema import CATEGORICAL_FEATURES, NUMERIC_FEATURES


@dataclass(frozen=True)
class Candidate:
    name: str
    pipeline: Pipeline


def linear_preprocessor(
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
) -> ColumnTransformer:
    numeric_columns = NUMERIC_FEATURES if numeric_features is None else numeric_features
    categorical_columns = (
        CATEGORICAL_FEATURES if categorical_features is None else categorical_features
    )
    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "one_hot",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=20,
                    sparse_output=True,
                ),
            ),
        ]
    )
    return ColumnTransformer(
        [("numeric", numeric, numeric_columns), ("categorical", categorical, categorical_columns)]
    )


def tree_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median", add_indicator=True))])
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "ordinal",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            ),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric, NUMERIC_FEATURES),
            ("categorical", categorical, CATEGORICAL_FEATURES),
        ],
        sparse_threshold=0,
    )


def candidates(seed: int) -> list[Candidate]:
    return [
        Candidate(
            "dummy_prior",
            Pipeline(
                [
                    ("preprocess", linear_preprocessor()),
                    ("model", DummyClassifier(strategy="prior")),
                ]
            ),
        ),
        Candidate(
            "logistic_regression",
            Pipeline(
                [
                    ("preprocess", linear_preprocessor()),
                    (
                        "model",
                        LogisticRegression(C=0.5, max_iter=1000, solver="lbfgs", random_state=seed),
                    ),
                ]
            ),
        ),
        Candidate(
            "hist_gradient_boosting",
            Pipeline(
                [
                    ("preprocess", tree_preprocessor()),
                    (
                        "model",
                        HistGradientBoostingClassifier(
                            learning_rate=0.06,
                            max_iter=180,
                            max_leaf_nodes=15,
                            min_samples_leaf=40,
                            l2_regularization=1.0,
                            early_stopping=True,
                            validation_fraction=0.15,
                            random_state=seed,
                        ),
                    ),
                ]
            ),
        ),
    ]
