from datetime import datetime
from typing import Any

from pydantic import BaseModel


class DetectionResponseDto(BaseModel):
    id: int
    frame_id: str
    camera_id: str
    object_key: str
    detections: dict[str, Any]
    created_at: datetime
