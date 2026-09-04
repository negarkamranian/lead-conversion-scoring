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
