from dataclasses import dataclass
from datetime import datetime

import pandas as pd
from pydantic import BaseModel

from lead_scoring.schema import TARGET, TIME_COLUMN


@dataclass(frozen=True)
class DataSplits:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    train_cutoff: datetime
    validation_cutoff: datetime


class SplitPartition(BaseModel):
    rows: int
    positives: int
    prevalence: float
    created_min: datetime
    created_max: datetime


class SplitSummary(BaseModel):
    strategy: str
    train_cutoff: datetime
    validation_cutoff: datetime
    train: SplitPartition
    validation: SplitPartition
    test: SplitPartition


def chronological_split(frame: pd.DataFrame) -> DataSplits:
    frame = frame.sort_values(TIME_COLUMN).reset_index(drop=True)

    train_cutoff = pd.Timestamp(frame[TIME_COLUMN].quantile(0.70, interpolation="nearest"))
    validation_cutoff = pd.Timestamp(frame[TIME_COLUMN].quantile(0.85, interpolation="nearest"))

    train = frame[frame[TIME_COLUMN] < train_cutoff]
    validation = frame[
        (frame[TIME_COLUMN] >= train_cutoff) & (frame[TIME_COLUMN] < validation_cutoff)
    ]
    test = frame[frame[TIME_COLUMN] >= validation_cutoff]

    if not all(len(part) for part in (train, validation, test)):
        raise ValueError("Chronological split produced an empty partition")

    return DataSplits(
        train=train,
        validation=validation,
        test=test,
        train_cutoff=train_cutoff,
        validation_cutoff=validation_cutoff,
    )


def split_summary(splits: DataSplits) -> SplitSummary:
    partitions: dict[str, SplitPartition] = {}

    for name in ("train", "validation", "test"):
        part = getattr(splits, name)

        partitions[name] = SplitPartition(
            rows=len(part),
            positives=int(part[TARGET].sum()),
            prevalence=float(part[TARGET].mean()),
            created_min=part[TIME_COLUMN].min(),
            created_max=part[TIME_COLUMN].max(),
        )

    return SplitSummary(
        strategy="chronological_70_15_15",
        train_cutoff=splits.train_cutoff,
        validation_cutoff=splits.validation_cutoff,
        train=partitions["train"],
        validation=partitions["validation"],
        test=partitions["test"],
    )
