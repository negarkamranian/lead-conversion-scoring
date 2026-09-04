from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from uuid import uuid4

import pandas as pd
import pytest

from lead_scoring.config import Settings
from lead_scoring.database import Database, ScoringBatch


@pytest.mark.integration
def test_postgres_ingestion_is_idempotent(raw_frame, tmp_path):
    settings = Settings()
    database = Database(settings)
    path = tmp_path / "fixture.csv"
    raw_frame.iloc[:8].to_csv(path, index=False)
    source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    batch_id = uuid4()
    database.initialize()
    first = database.ingest_csv(path)
    second = database.ingest_csv(path)
    assert first == second == source_hash
    with database.connect() as conn:
        batch_count = conn.execute(
            "SELECT COUNT(*) FROM ingestion_batches WHERE source_hash = %s", (source_hash,)
        ).fetchone()[0]
        row_count = conn.execute(
            "SELECT COUNT(*) FROM raw_leads WHERE source_hash = %s", (source_hash,)
        ).fetchone()[0]
    assert batch_count == 1
    assert row_count == 8
    now = datetime.now(UTC)
    scores = pd.DataFrame(
        {
            "scoring_batch_id": [batch_id, batch_id],
            "lead_id": ["fixture-L1", "fixture-L2"],
            "purchase_probability": [0.8, 0.4],
            "priority_rank": [1, 2],
            "priority_tier": ["call", "backlog"],
            "scored_at": [now, now],
            "model_version": ["test-v1", "test-v1"],
            "data_as_of": [now, now],
        }
    )
    batch = ScoringBatch(
        scoring_batch_id=batch_id,
        model_version="test-v1",
        source_hash=source_hash,
        data_as_of=now,
        started_at=now,
        completed_at=now,
    )
    try:
        database.write_scores(scores, batch)
        database.write_scores(scores, batch)
        with database.connect() as conn:
            assert (
                conn.execute(
                    "SELECT COUNT(*) FROM scoring_batches WHERE scoring_batch_id = %s", (batch_id,)
                ).fetchone()[0]
                == 1
            )
            assert (
                conn.execute(
                    "SELECT COUNT(*) FROM lead_scores WHERE scoring_batch_id = %s", (batch_id,)
                ).fetchone()[0]
                == 2
            )
            missing_lineage = conn.execute(
                "SELECT COUNT(*) FROM lead_scores WHERE scoring_batch_id = %s "
                "AND (model_version IS NULL OR data_as_of IS NULL OR scored_at IS NULL)",
                (batch_id,),
            ).fetchone()[0]
            assert missing_lineage == 0
    finally:
        with database.connect() as conn:
            conn.execute("DELETE FROM lead_scores WHERE scoring_batch_id = %s", (batch_id,))
            conn.execute("DELETE FROM scoring_batches WHERE scoring_batch_id = %s", (batch_id,))
            conn.execute("DELETE FROM raw_leads WHERE source_hash = %s", (source_hash,))
            conn.execute("DELETE FROM ingestion_batches WHERE source_hash = %s", (source_hash,))
