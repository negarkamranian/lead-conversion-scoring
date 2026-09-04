import json
import logging
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
from pydantic import BaseModel

from lead_scoring.data.audit import feature_availability_audit
from lead_scoring.data.preparation import RAW_NUMERIC, prepare_data
from lead_scoring.schema import (
    CATEGORICAL_FEATURES,
    ID_COLUMN,
    TARGET,
    TIME_COLUMN,
)

logger = logging.getLogger(__name__)


class QualityReport(BaseModel):
    """Structured output produced by the data-quality workflow."""

    rows_raw: int
    rows_clean: int
    columns: int
    missing_counts: dict[str, int]
    duplicate_rows: int
    duplicate_lead_ids: int
    target_prevalence: float
    date_min: datetime
    date_max: datetime
    cardinality: dict[str, int]
    rare_categories: dict[str, list[str]]
    outliers: dict[str, int]


def outlier_counts(frame):
    counts = {}

    for column in RAW_NUMERIC:
        values = frame[column].dropna()
        q1, q3 = values.quantile([0.25, 0.75])
        iqr = q3 - q1

        counts[column] = ((values < q1 - 3 * iqr) | (values > q3 + 3 * iqr)).sum()

    return counts


def rare_categories(frame):
    minimum = max(20, int(len(frame) * 0.001))

    return {
        column: frame[column]
        .value_counts(dropna=False)
        .loc[lambda x: x < minimum]
        .index.astype(str)
        .tolist()
        for column in CATEGORICAL_FEATURES
    }


def build_quality_report(raw) -> tuple[QualityReport, pd.DataFrame]:
    frame = prepare_data(raw).frame

    report = QualityReport(
        rows_raw=len(raw),
        rows_clean=len(frame),
        columns=len(raw.columns),
        missing_counts=raw.replace("", pd.NA).isna().sum().to_dict(),
        duplicate_rows=raw.duplicated().sum(),
        duplicate_lead_ids=raw.duplicated(ID_COLUMN).sum(),
        target_prevalence=frame[TARGET].mean(),
        date_min=frame[TIME_COLUMN].min(),
        date_max=frame[TIME_COLUMN].max(),
        cardinality={column: frame[column].nunique() for column in CATEGORICAL_FEATURES},
        rare_categories=rare_categories(frame),
        outliers=outlier_counts(frame),
    )

    return report, frame


def generate_eda_charts(frame, chart_dir):
    monthly = (
        frame.assign(month=frame[TIME_COLUMN].dt.strftime("%Y-%m"))
        .groupby("month")[TARGET]
        .agg(["count", "mean"])
    )

    figure, axis = plt.subplots(figsize=(8, 4))

    axis.plot(monthly.index, monthly["mean"], marker="o")
    axis.set(
        xlabel="Month",
        ylabel="Purchase rate",
        title="Purchase rate over time",
    )

    figure.tight_layout()
    figure.savefig(chart_dir / "monthly_purchase_rate.png")
    plt.close(figure)

    abandonment = (
        frame.assign(
            bucket=pd.qcut(
                frame["Minutes Since Abandonment"],
                10,
                duplicates="drop",
            )
        )
        .groupby("bucket", observed=True)[TARGET]
        .mean()
    )

    figure, axis = plt.subplots(figsize=(8, 4))

    axis.plot(
        range(1, len(abandonment) + 1),
        abandonment,
        marker="o",
    )

    axis.set(
        xlabel="Abandonment decile",
        ylabel="Purchase rate",
        title="Purchase rate by abandonment time",
    )

    figure.tight_layout()
    figure.savefig(chart_dir / "abandonment_conversion.png")
    plt.close(figure)


def run_quality(raw, artifact_dir, chart_dir):
    report, frame = build_quality_report(raw)

    with open(artifact_dir / "data_quality_report.json", "w") as file:
        json.dump(report.model_dump(mode="json"), file)

    feature_availability_audit().to_csv(
        artifact_dir / "feature_availability_audit.csv",
        index=False,
    )

    generate_eda_charts(frame, chart_dir)

    logger.info("Data quality analysis completed")

    return report
