from __future__ import annotations

from unittest.mock import Mock

import psycopg

from lead_scoring.config import Settings
from lead_scoring.database import Database


def test_database_connection_retries_only_connection_failures(monkeypatch):
    connection = Mock()
    connect = Mock(
        side_effect=[
            psycopg.OperationalError("temporarily unavailable"),
            connection,
        ]
    )
    sleep = Mock()
    monkeypatch.setattr("lead_scoring.database.psycopg.connect", connect)
    monkeypatch.setattr("lead_scoring.database.time.sleep", sleep)

    with Database(Settings()).connect() as opened:
        assert opened is connection

    assert connect.call_count == 2
    assert sleep.call_count == 1
    connection.commit.assert_called_once_with()
    connection.close.assert_called_once_with()


def test_database_transaction_rolls_back_without_retry(monkeypatch):
    connection = Mock()
    connect = Mock(return_value=connection)
    monkeypatch.setattr("lead_scoring.database.psycopg.connect", connect)

    try:
        with Database(Settings()).connect():
            raise ValueError("bad statement")
    except ValueError:
        pass

    connect.assert_called_once()
    connection.rollback.assert_called_once_with()
    connection.commit.assert_not_called()
    connection.close.assert_called_once_with()


def test_write_scores_preserves_column_order(monkeypatch):
    from datetime import UTC, datetime
    from unittest.mock import MagicMock
    from uuid import uuid4

    import pandas as pd

    from lead_scoring.database import ScoringBatch

    now = datetime.now(UTC)
    batch = ScoringBatch(
        scoring_batch_id=uuid4(),
        model_version="v1",
        source_hash="source",
        data_as_of=now,
        started_at=now,
        completed_at=now,
    )
    record = {
        "lead_id": "lead-1",
        "purchase_probability": 0.8,
        "priority_rank": 1,
        "priority_tier": "call",
        "scored_at": now,
        "model_version": batch.model_version,
        "data_as_of": now,
        "scoring_batch_id": batch.scoring_batch_id,
    }
    connection = MagicMock()
    connection.execute.return_value.fetchone.return_value = None
    cursor = connection.cursor.return_value.__enter__.return_value
    written = []
    cursor.executemany.side_effect = lambda query, rows: written.extend(rows)
    database = Database(Settings())
    monkeypatch.setattr(database, "_connect_with_retry", lambda: connection)

    database.write_scores(pd.DataFrame([record]), batch)

    assert written == [(batch.scoring_batch_id, "lead-1", 0.8, 1, "call", now, "v1", now)]
    connection.commit.assert_called_once()
    connection.close.assert_called_once()
