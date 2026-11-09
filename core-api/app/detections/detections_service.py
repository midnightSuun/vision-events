from typing import Annotated, Any, cast

from fastapi import Depends

from app.detections.detections_repository import DetectionsRepository
from app.detections.dto.detection_response_dto import DetectionResponseDto
from app.models import Detections


class DetectionsService:
    def __init__(self, detections_repository: Annotated[DetectionsRepository, Depends()]):
        self.detections_repository = detections_repository

    async def get_detections(self, limit: int, camera_id: str | None) -> list[DetectionResponseDto]:
        rows = await self.detections_repository.get_detections(limit, camera_id)
        return [self._to_response(row) for row in rows]

    def _to_response(self, row: Detections) -> DetectionResponseDto:
        return DetectionResponseDto(
            id=row.id,
            frame_id=row.frame_id,
            camera_id=row.camera_id,
            object_key=row.object_key,
            detections=cast(dict[str, Any], row.detections),
            created_at=row.created_at,
        )
