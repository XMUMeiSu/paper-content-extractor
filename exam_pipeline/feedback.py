"""Read-only HITL geometry feedback."""
import sqlite3
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class GeometryFeedbackProfile:
    scope: str
    delta_x: float
    delta_y: float
    delta_w: float
    delta_h: float
    samples: int


class HITLFeedbackReader:
    def __init__(self, database: Path):
        self.database = Path(database)

    def profile(self, item_id: str) -> Optional[GeometryFeedbackProfile]:
        if not self.database.is_file():
            return None
        try:
            uri = f"file:{self.database.resolve()}?mode=ro"
            with sqlite3.connect(uri, uri=True) as connection:
                rows = connection.execute(
                    "SELECT delta_x, delta_y, delta_w, delta_h FROM hitl_audit_events WHERE item_id=?",
                    (item_id,),
                ).fetchall()
        except sqlite3.Error:
            return None
        if not rows:
            return None
        medians = [float(statistics.median(column)) for column in zip(*rows)]
        return GeometryFeedbackProfile(f"item:{item_id}", *medians, len(rows))

    @staticmethod
    def apply(box: List[float], profile: GeometryFeedbackProfile,
              max_offset: float = 80.0) -> List[float]:
        x1, y1, x2, y2 = [float(value) for value in box[:4]]
        dx = max(-max_offset, min(max_offset, profile.delta_x))
        dy = max(-max_offset, min(max_offset, profile.delta_y))
        dw = max(-max_offset, min(max_offset, profile.delta_w))
        dh = max(-max_offset, min(max_offset, profile.delta_h))
        return [x1 + dx, y1 + dy, x2 + dx + dw, y2 + dy + dh]
