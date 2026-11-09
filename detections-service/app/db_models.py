"""Generated from the live Postgres schema. Do not edit by hand.

Regenerate after a migration:

    uv run python scripts/generate_models.py
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel


class Detections(BaseModel):
    id: int
    frame_id: str
    camera_id: str
    object_key: str
    detections: dict[str, Any]
    created_at: datetime
