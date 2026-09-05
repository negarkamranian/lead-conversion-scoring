from __future__ import annotations

import hashlib
import logging
import random
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated, Self
from uuid import UUID

import pandas as pd
import psycopg
from psycopg import Connection
from pydantic import AwareDatetime, BaseModel, ConfigDict, StringConstraints, model_validator

from lead_scoring.config import Settings
from lead_scoring.schema import CSV_COLUMNS, DB_COLUMNS

LOGGER = logging.getLogger(__name__)
MAX_CONNECTION_ATTEMPTS = 3
SCORE_COLUMNS = [
    "scoring_batch_id",
    "lead_id",
    "purchase_probability",
    "priority_rank",
    "priority_tier",
    "scored_at",
    "model_version",
    "data_as_of",
]


class ScoringBatch(BaseModel):
    """Lineage recorded once for an immutable scoring run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    scoring_batch_id: UUID
    model_version: Annotated[str, StringConstraints(min_length=1)]
    source_hash: Annotated[str, StringConstraints(min_length=1)]
    data_as_of: AwareDatetime
    started_at: AwareDatetime
    completed_at: AwareDatetime

    @model_validator(mode="after")
    def validate_timing(self) -> Self:
        if self.completed_at < self.started_at:
            raise ValueError("completed_at cannot precede started_at")
        return self


class Database:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def _connect_with_retry(self) -> Connection:
        attempt = 1
        while True:
            try:
                return psycopg.connect(
                    dbname=self.settings.postgres_db,
                    user=self.settings.postgres_user,
                    password=self.settings.postgres_password,
                    host=self.settings.db_host,
                    port=self.settings.db_port,
                )
            except psycopg.OperationalError as exc:
                if attempt == MAX_CONNECTION_ATTEMPTS:
                    raise RuntimeError
                delay = 0.1 * (2 ** (attempt - 1)) + random.uniform(0, 0.05)
                LOGGER.warning(
                    "PostgreSQL connection attempt %d/%d failed; retrying in %.2fs",
                    attempt,
                    MAX_CONNECTION_ATTEMPTS,
                    delay,
                )
                time.sleep(delay)
                attempt += 1

    @contextmanager
    def connect(self) -> Iterator[Connection]:
        """Provide one transaction and always commit/rollback and close it."""
        conn = self._connect_with_retry()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def initialize(self, sql_path: Path = Path("sql/001_init.sql")) -> None:
        with self.connect() as conn:
            conn.execute(sql_path.read_text(encoding="utf-8"))
        LOGGER.info("Database schema initialized")

    def ingest_csv(self, csv_path: Path, dictionary_path: Path | None = None) -> str:
        source_hash = hashlib.sha256(csv_path.read_bytes()).hexdigest()
        raw = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
        if list(raw.columns) != CSV_COLUMNS:
            missing = sorted(set(CSV_COLUMNS) - set(raw.columns))
            extra = sorted(set(raw.columns) - set(CSV_COLUMNS))
            raise ValueError(f"Unexpected CSV schema; missing={missing}, extra={extra}")

        with self.connect() as conn:
            inserted = conn.execute(
                """
                INSERT INTO ingestion_batches(source_hash, source_name, row_count)
                VALUES (%s, %s, %s)
                ON CONFLICT (source_hash) DO NOTHING
                RETURNING source_hash
                """,
                (source_hash, csv_path.name, len(raw)),
            ).fetchone()
            if not inserted:
                LOGGER.info("Source %s already ingested; skipping", source_hash[:12])
                self._upsert_dictionary(conn, dictionary_path)
                return source_hash
            columns = ["source_hash", "source_row_number", *DB_COLUMNS]
            copy_sql = f"COPY raw_leads ({', '.join(columns)}) FROM STDIN"
            with conn.cursor().copy(copy_sql) as copy:
                for row_number, values in enumerate(raw.itertuples(index=False, name=None), 1):
                    copy.write_row((source_hash, row_number, *values))
            self._upsert_dictionary(conn, dictionary_path)
        LOGGER.info("Ingested %d raw rows with source hash %s", len(raw), source_hash[:12])
        return source_hash

    @staticmethod
    def _upsert_dictionary(conn: Connection, path: Path | None) -> None:
        if path is None:
            return
        dictionary = pd.read_csv(path, encoding="utf-8-sig", dtype=str)
        with conn.cursor() as cursor:
            cursor.executemany(
                """
                INSERT INTO data_dictionary(column_name, description)
                VALUES (%s, %s)
                ON CONFLICT (column_name) DO UPDATE
                SET description = EXCLUDED.description, loaded_at = NOW()
                """,
                list(dictionary[["column_name", "description"]].itertuples(index=False, name=None)),
            )

    def latest_source_hash(self) -> str:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT source_hash FROM ingestion_batches
                ORDER BY ingested_at DESC, source_hash DESC
                LIMIT 1
                """
            ).fetchone()
        if not row:
            raise RuntimeError("No ingested data found; run ingest first")
        return str(row[0])

    def load_raw(self, source_hash: str | None = None) -> tuple[pd.DataFrame, str]:
        selected_hash = source_hash or self.latest_source_hash()
        select_columns = ", ".join(DB_COLUMNS)
        query = f"""
            SELECT {select_columns}
            FROM raw_leads
            WHERE source_hash = %s
            ORDER BY source_row_number
        """
        with self.connect() as conn:
            rows = conn.execute(query, (selected_hash,)).fetchall()
        if not rows:
            raise RuntimeError(f"No raw rows for source hash {selected_hash}")
        return pd.DataFrame(rows, columns=CSV_COLUMNS), selected_hash

    def write_scores(self, scores: pd.DataFrame, batch: ScoringBatch) -> None:
        with self.connect() as conn:
            existing = conn.execute(
                """
                SELECT status, row_count, model_version, source_hash
                FROM scoring_batches
                WHERE scoring_batch_id = %s
                """,
                (batch.scoring_batch_id,),
            ).fetchone()
            if existing:
                expected = ("succeeded", len(scores), batch.model_version, batch.source_hash)
                if tuple(existing) != expected:
                    raise ValueError(
                        f"Scoring batch {batch.scoring_batch_id} exists with different lineage: "
                        f"stored={tuple(existing)!r}, requested={expected!r}"
                    )
                LOGGER.info("Scoring batch %s already exists; skipping", batch.scoring_batch_id)
                return
            conn.execute(
                """
                INSERT INTO scoring_batches(
                    scoring_batch_id, model_version, source_hash,
                    data_as_of, started_at, completed_at, status, row_count
                )
                VALUES (%s, %s, %s, %s, %s, %s, 'succeeded', %s)
                """,
                (
                    batch.scoring_batch_id,
                    batch.model_version,
                    batch.source_hash,
                    batch.data_as_of,
                    batch.started_at,
                    batch.completed_at,
                    len(scores),
                ),
            )
            rows = scores[SCORE_COLUMNS].itertuples(index=False, name=None)
            with conn.cursor() as cursor:
                cursor.executemany(
                    f"""
                    INSERT INTO lead_scores({", ".join(SCORE_COLUMNS)})
                    VALUES ({", ".join(["%s"] * len(SCORE_COLUMNS))})
                    """,
                    rows,
                )
        LOGGER.info("Stored %d scores for batch %s", len(scores), batch.scoring_batch_id)

    def load_latest_scores(self) -> pd.DataFrame:
        query = """
            SELECT s.lead_id, s.purchase_probability, s.priority_rank, s.priority_tier,
                   s.scored_at, s.model_version, s.scoring_batch_id, s.data_as_of
            FROM lead_scores s
            JOIN scoring_batches b USING (scoring_batch_id)
            WHERE b.status = 'succeeded'
              AND b.scoring_batch_id = (
                  SELECT scoring_batch_id FROM scoring_batches
                  WHERE status = 'succeeded'
                  ORDER BY completed_at DESC, scoring_batch_id DESC
                  LIMIT 1
              )
            ORDER BY s.priority_rank
        """
        with self.connect() as conn:
            rows = conn.execute(query).fetchall()
        if not rows:
            raise RuntimeError("No successful scoring batch found; run score first")
        columns = [
            "lead_id",
            "purchase_probability",
            "priority_rank",
            "priority_tier",
            "scored_at",
            "model_version",
            "scoring_batch_id",
            "data_as_of",
        ]
        return pd.DataFrame(rows, columns=columns)
