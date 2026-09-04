from __future__ import annotations

from lead_scoring.config import Settings
from lead_scoring.data.quality import run_quality
from lead_scoring.database import Database
from lead_scoring.monitoring import monitor
from lead_scoring.scoring import score
from lead_scoring.training import evaluate, train

COMMAND_NAMES = (
    "init-db",
    "ingest",
    "quality",
    "train",
    "evaluate",
    "score",
    "monitor",
    "pipeline",
)


def run_command(command: str, settings: Settings, database: Database) -> None:
    """Run one application workflow selected by the CLI boundary."""
    if command == "init-db":
        database.initialize()
        return
    if command == "ingest":
        database.initialize()
        database.ingest_csv(settings.data_path, settings.dictionary_path)
        return

    if command == "pipeline":
        database.initialize()
        source_hash = database.ingest_csv(settings.data_path, settings.dictionary_path)
        raw, _ = database.load_raw(source_hash)
        run_quality(raw, settings.artifact_dir, settings.chart_dir)
        train(raw, source_hash, settings)
        evaluate(raw, source_hash, settings)
        score(raw, source_hash, settings, database)
        monitor(raw, database.load_latest_scores(), settings.artifact_dir)
        return

    raw, source_hash = database.load_raw()
    if command == "quality":
        run_quality(raw, settings.artifact_dir, settings.chart_dir)
    elif command == "train":
        train(raw, source_hash, settings)
    elif command == "evaluate":
        evaluate(raw, source_hash, settings)
    elif command == "score":
        score(raw, source_hash, settings, database)
    elif command == "monitor":
        monitor(raw, database.load_latest_scores(), settings.artifact_dir)
    else:
        raise ValueError(f"Unknown command: {command}")
