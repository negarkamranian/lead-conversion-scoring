from __future__ import annotations

from dataclasses import dataclass

from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, SplineTransformer, StandardScaler
from xgboost import XGBClassifier

from lead_scoring.schema import CATEGORICAL_FEATURES, NUMERIC_FEATURES


@dataclass(frozen=True)
class Candidate:
    name: str
    pipeline: Pipeline


def linear_preprocessor(
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
    *,
    splines: bool = False,
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
    if splines:
        numeric.steps.insert(
            1, ("splines", SplineTransformer(n_knots=4, degree=3, include_bias=False))
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


def catboost_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("numeric", "passthrough", NUMERIC_FEATURES),
            (
                "categorical",
                SimpleImputer(strategy="constant", fill_value="missing"),
                CATEGORICAL_FEATURES,
            ),
        ],
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")


def candidates(seed: int) -> list[Candidate]:
    specifications = [
        ("dummy_prior", linear_preprocessor(), DummyClassifier(strategy="prior")),
        (
            "logistic_regression",
            linear_preprocessor(),
            LogisticRegression(C=0.5, max_iter=1000, solver="lbfgs", random_state=seed),
        ),
        (
            "spline_logistic_regression",
            linear_preprocessor(splines=True),
            LogisticRegression(C=0.5, max_iter=1000, random_state=seed),
        ),
        (
            "extra_trees",
            linear_preprocessor(),
            ExtraTreesClassifier(
                n_estimators=300,
                min_samples_leaf=20,
                max_features=0.8,
                n_jobs=-1,
                random_state=seed,
            ),
        ),
        (
            "random_forest",
            tree_preprocessor(),
            RandomForestClassifier(
                n_estimators=300,
                min_samples_leaf=20,
                max_features=0.8,
                n_jobs=-1,
                random_state=seed,
            ),
        ),
        (
            "gradient_boosting",
            tree_preprocessor(),
            GradientBoostingClassifier(
                n_estimators=150,
                learning_rate=0.04,
                max_depth=2,
                min_samples_leaf=40,
                subsample=0.8,
                random_state=seed,
            ),
        ),
        (
            "hist_gradient_boosting",
            tree_preprocessor(),
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
        (
            "catboost_depth_4",
            catboost_preprocessor(),
            CatBoostClassifier(
                iterations=300,
                depth=4,
                learning_rate=0.04,
                l2_leaf_reg=10,
                cat_features=CATEGORICAL_FEATURES,
                loss_function="Logloss",
                thread_count=2,
                random_seed=seed,
                verbose=False,
                allow_writing_files=False,
            ),
        ),
        (
            "xgboost_depth_4",
            linear_preprocessor(),
            XGBClassifier(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.04,
                min_child_weight=20,
                reg_lambda=10,
                subsample=0.8,
                colsample_bytree=0.8,
                tree_method="hist",
                objective="binary:logistic",
                eval_metric="logloss",
                n_jobs=2,
                random_state=seed,
            ),
        ),
    ]
    return [
        Candidate(name, Pipeline([("preprocess", preprocess), ("model", model)]))
        for name, preprocess, model in specifications
    ]
