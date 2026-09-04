from __future__ import annotations

import pytest
from pydantic import ValidationError

from lead_scoring.config import Settings


@pytest.mark.parametrize(
    "values",
    [
        {"db_port": "not-an-integer"},
        {"db_port": "70000"},
        {"top_fraction": "0"},
        {"postgres_password": ""},
        {"data_path": ""},
    ],
)
def test_settings_reject_invalid_values(values):
    with pytest.raises(ValidationError):
        Settings.model_validate(values)


def test_settings_read_environment(monkeypatch):
    monkeypatch.setenv("DB_PORT", "5544")
    monkeypatch.setenv("TOP_FRACTION", "0.25")
    monkeypatch.setenv("RANDOM_SEED", "7")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    settings = Settings(_env_file=None)
    assert settings.db_port == 5544
    assert settings.top_fraction == 0.25
    assert settings.random_seed == 7
    assert settings.log_level == "DEBUG"
