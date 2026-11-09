from typing import Annotated

from fastapi import APIRouter, Depends, Query

from app.detections.detections_service import DetectionsService
from app.detections.dto.detection_response_dto import DetectionResponseDto

router = APIRouter(prefix="/detections", tags=["detections"])


@router.get("")
async def get_detections(
    detections_service: Annotated[DetectionsService, Depends()],
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    camera_id: str | None = None,
) -> list[DetectionResponseDto]:
    return await detections_service.get_detections(limit, camera_id)
