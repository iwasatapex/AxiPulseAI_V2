
import json
from pathlib import Path
from typing import List, Dict, Any


class HistoryStore:
    """
    Phase F1:
    Persistent forecast memory.
    """

    def __init__(self, path="data/forecast_history.json"):
        self.path = Path(path)
        self.path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

    def load(self) -> Dict[str, Any]:
        if not self.path.exists():
            return {
                "forecast_history": [],
                "actual_history": [],
                "forecast_error_history": []
            }

        return json.loads(
            self.path.read_text()
        )

    def save(self, data: Dict[str, Any]):
        self.path.write_text(
            json.dumps(
                data,
                indent=2,
                default=str
            )
        )

    def append(
        self,
        section: str,
        record: Dict[str, Any]
    ):
        data = self.load()

        if section not in data:
            data[section] = []

        data[section].append(record)

        self.save(data)
