from __future__ import annotations

import argparse
import logging

from lead_scoring.config import Settings
from lead_scoring.database import Database
from lead_scoring.workflow import COMMAND_NAMES, run_command

logger = logging.getLogger(__name__)


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Leakage-aware lead prioritization pipeline")
    parser.add_argument("command", choices=COMMAND_NAMES)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    settings = Settings()
    configure_logging(settings.log_level)
    settings.ensure_output_dirs()

    logger.info("Starting %s", args.command)
    run_command(args.command, settings, Database(settings))
    logger.info("Completed %s", args.command)


if __name__ == "__main__":
    main()
