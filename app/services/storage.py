import json
from pathlib import Path
from typing import Any


class JsonStore:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.write([])

    def read(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        return json.loads(self.path.read_text(encoding="utf-8"))

    def write(self, payload: list[dict[str, Any]]) -> None:
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
