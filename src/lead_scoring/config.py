from pathlib import Path
from typing import Annotated

from pydantic import Field, FilePath, StringConstraints
from pydantic_settings import BaseSettings, SettingsConfigDict

NonEmptyString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Port = Annotated[int, Field(ge=1, le=65535)]
Fraction = Annotated[float, Field(gt=0, le=1)]
Seed = Annotated[int, Field(ge=0)]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

    postgres_db: NonEmptyString = "lead_scoring"
    postgres_user: NonEmptyString = "lead_app"
    postgres_password: NonEmptyString = "local_dev_only"
    db_host: NonEmptyString = "localhost"
    db_port: Port = 5432

    data_path: FilePath = Path("data/leads.csv")
    dictionary_path: FilePath = Path("data/data_dictionary.csv")
    artifact_dir: Path = Path("artifacts")
    chart_dir: Path = Path("charts")

    random_seed: Seed = 42
    top_fraction: Fraction = 0.1

    def ensure_output_dirs(self) -> None:
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.chart_dir.mkdir(parents=True, exist_ok=True)
