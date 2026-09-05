import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel


def write_json(path: Path, payload: BaseModel | dict[str, Any]) -> None:
    content = (
        payload.model_dump_json(indent=2)
        if isinstance(payload, BaseModel)
        else json.dumps(payload, indent=2)
    )
    path.write_text(content, encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
