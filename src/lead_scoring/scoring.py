import logging
from datetime import UTC, datetime
from uuid import uuid4

import pandas as pd

from lead_scoring.artifacts import load_model_bundle
from lead_scoring.data.preparation import prepare_data, select_features
from lead_scoring.database import ScoringBatch
from lead_scoring.metrics import capacity_count
from lead_scoring.schema import ID_COLUMN, TIME_COLUMN

logger = logging.getLogger(__name__)


def rank_leads(lead_ids, probabilities, capacity_fraction):
    frame = pd.DataFrame(
        {
            "lead_id": lead_ids.astype(str),
            "purchase_probability": probabilities,
        }
    )

    frame = frame.sort_values(
        ["purchase_probability", "lead_id"],
        ascending=[False, True],
    ).reset_index(drop=True)

    frame["priority_rank"] = range(1, len(frame) + 1)

    call_capacity = capacity_count(len(frame), capacity_fraction)
    frame["priority_tier"] = "backlog"
    frame.loc[: call_capacity - 1, "priority_tier"] = "call"

    return frame


def score(raw, source_hash, settings, database, batch_id=None):
    bundle = load_model_bundle(settings.artifact_dir)
    prepared = prepare_data(raw, require_target=False)

    probabilities = bundle.pipeline.predict_proba(select_features(prepared.frame))[:, 1]

    ranked = rank_leads(
        prepared.frame[ID_COLUMN],
        probabilities,
        bundle.top_fraction,
    )

    now = datetime.now(UTC)
    batch_id = batch_id or uuid4()
    data_as_of = prepared.frame[TIME_COLUMN].max().to_pydatetime()

    ranked["scored_at"] = now
    ranked["model_version"] = bundle.model_version
    ranked["scoring_batch_id"] = batch_id
    ranked["data_as_of"] = data_as_of

    batch = ScoringBatch(
        scoring_batch_id=batch_id,
        model_version=bundle.model_version,
        source_hash=source_hash,
        data_as_of=data_as_of,
        started_at=now,
        completed_at=datetime.now(UTC),
    )

    database.write_scores(ranked, batch)

    logger.info("Scored %d leads", len(ranked))

    return ranked
