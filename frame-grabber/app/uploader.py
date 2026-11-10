"""Put one JPEG into the frames bucket."""

import logging
import time
from io import BytesIO

from minio import Minio
from minio.error import S3Error

logger = logging.getLogger("frame-grabber")

_ATTEMPTS = 3


class FrameUploader:
    def __init__(self, client: Minio, bucket: str) -> None:
        self._client = client
        self._bucket = bucket

    def put_jpeg(self, camera_id: str, frame_id: str, payload: bytes) -> bool:
        key = f"{camera_id}/{frame_id}.jpg"
        delay = 0.5
        for attempt in range(1, _ATTEMPTS + 1):
            try:
                self._client.put_object(
                    self._bucket,
                    key,
                    BytesIO(payload),
                    length=len(payload),
                    content_type="image/jpeg",
                )
            except S3Error as exc:
                logger.warning("upload %s failed (%s/%s): %s", key, attempt, _ATTEMPTS, exc)
                if attempt == _ATTEMPTS:
                    return False
                time.sleep(delay)
                delay *= 2
                continue
            logger.info("stored %s/%s (%s bytes)", self._bucket, key, len(payload))
            return True
        return False
