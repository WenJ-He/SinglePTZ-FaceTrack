"""Event logger: append recognition results to JSONL."""

import json
import logging
import os
from datetime import datetime
from typing import Optional

logger = logging.getLogger("app")


class EventLogger:
    """Append recognition events to events.jsonl."""

    def __init__(self, path: str = "output/events.jsonl"):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._count = 0

    def log(self, result_kind: str, name: Optional[str],
            sim: float, preset_id: Optional[int] = None,
            track_id: Optional[int] = None,
            bbox: Optional[tuple] = None,
            snapshot: Optional[str] = None):
        """Append one event to JSONL file."""
        self._count += 1
        event = {
            "ts": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            "preset": preset_id,
            "track_id": track_id,
            "event_id": self._count,
            "result": result_kind,
            "name": name,
            "sim": round(sim, 4),
        }
        if bbox:
            event["bbox"] = list(bbox)
        if snapshot:
            event["snapshot"] = snapshot

        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

        logger.info(
            f"Event #{self._count}: {result_kind} "
            f"name={name} sim={sim:.4f}"
        )
