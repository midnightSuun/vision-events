from typing import Annotated

from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse

from app.detections.consumer import hub, stream_detections
from app.detections.detections_service import DetectionService
from app.detections.dto.detection_response_dto import DetectionResponseDto

router = APIRouter(tags=["detections"])

_STREAM_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


@router.get("/detections")
async def get_detections(
    detections_service: Annotated[DetectionService, Depends()],
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    camera_id: str | None = None,
) -> list[DetectionResponseDto]:
    return await detections_service.get_detections(limit, camera_id)


@router.get("/detections/stream")
async def get_detection_stream() -> StreamingResponse:
    queue = hub.subscribe()
    return StreamingResponse(
        stream_detections(queue),
        media_type="text/event-stream",
        headers=_STREAM_HEADERS,
    )
