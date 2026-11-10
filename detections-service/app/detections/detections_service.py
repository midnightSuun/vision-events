import asyncio
import logging
import time
from collections.abc import Callable

from minio import Minio
from sqlalchemy.exc import SQLAlchemyError

from app.detections.detections_repository import DetectionsRepository
from app.detections.frames import FrameRef, prepare_frame
from infer import Classifier

logger = logging.getLogger("detection-service")


async def _sleep_while(should_continue: Callable[[], bool], seconds: float) -> bool:
    deadline = time.monotonic() + seconds
    while should_continue() and time.monotonic() < deadline:
        await asyncio.sleep(0.2)
    return should_continue()


class DetectionService:
    def __init__(
        self,
        detections_repository: DetectionsRepository,
        client: Minio,
        classifier: Classifier,
    ):
        self.detections_repository = detections_repository
        self.client = client
        self.classifier = classifier

    async def process_frame(self, frame: FrameRef, should_continue: Callable[[], bool]) -> bool:
        """Returns False if the service stopped before the row was stored."""
        prepared = await asyncio.to_thread(prepare_frame, frame, self.client, self.classifier)
        if prepared is None:
            return True

        delay = 1.0
        while should_continue():
            try:
                detection_id = await self.detections_repository.insert_detection(
                    prepared.frame_id,
                    prepared.camera_id,
                    frame.key,
                    prepared.payload,
                )
            except (SQLAlchemyError, RuntimeError) as exc:
                logger.error("failed to store %s/%s: %s", frame.bucket, frame.key, exc)
                if not await _sleep_while(should_continue, delay):
                    return False
                delay = min(delay * 2, 30.0)
                continue
            logger.info("stored detection %s for %s/%s", detection_id, frame.bucket, frame.key)
            return True
        return False
