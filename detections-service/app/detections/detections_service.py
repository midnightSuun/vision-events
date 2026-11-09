"""Frame from MinIO, model class and box, then a row in Postgres."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple
from urllib.parse import unquote_plus

from minio import Minio
from minio.error import S3Error
from PIL import UnidentifiedImageError
from sqlalchemy.exc import SQLAlchemyError

from app.db import AsyncSessionLocal
from app.detections.detections_repository import DetectionsRepository
from app.settings import Settings, get_settings
from infer import Classifier, FramePrediction

logger = logging.getLogger("detection-service")


class FrameRef(NamedTuple):
    bucket: str
    key: str


class PreparedFrame(NamedTuple):
    camera_id: str
    frame_id: str
    payload: dict[str, object]


_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")


def _is_image(key: str) -> bool:
    return key.lower().endswith(_IMAGE_SUFFIXES)


def frame_refs(raw: bytes | None) -> list[FrameRef]:
    """Reads bucket and object.key from a MinIO notification."""
    if not raw:
        return []
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError):
        logger.warning("message is not json, skipping")
        return []
    if not isinstance(payload, dict):
        return []

    refs: list[FrameRef] = []
    records = payload.get("Records")
    if isinstance(records, list):
        for record in records:
            ref = _record_ref(record)
            if ref is not None:
                refs.append(ref)
    if refs:
        return refs

    return _key_field_refs(payload.get("Key"))


def _record_ref(record: object) -> FrameRef | None:
    if not isinstance(record, dict):
        return None
    s3 = record.get("s3")
    if not isinstance(s3, dict):
        return None
    bucket_info = s3.get("bucket")
    obj = s3.get("object")
    if not isinstance(bucket_info, dict) or not isinstance(obj, dict):
        return None
    bucket = bucket_info.get("name")
    key = obj.get("key")
    if not isinstance(bucket, str) or not bucket:
        return None
    if not isinstance(key, str) or not key:
        return None
    return FrameRef(unquote_plus(bucket), unquote_plus(key))


def _key_field_refs(raw_key: object) -> list[FrameRef]:
    """Fallback MinIO format: Key = bucket/object-key."""
    if not isinstance(raw_key, str) or "/" not in raw_key:
        return []
    bucket, key = raw_key.split("/", 1)
    bucket = unquote_plus(bucket)
    key = unquote_plus(key)
    if not bucket or not key:
        return []
    return [FrameRef(bucket, key)]


def frame_identity(key: str) -> tuple[str, str]:
    """The first directory of the key is camera_id; the filename without its extension is frame_id.

    cam-1/frame-9.jpg → camera cam-1, frame frame-9.
    frame-9.jpg → camera unknown, frame frame-9.
    """
    parts = [part for part in key.split("/") if part]
    if not parts:
        return "unknown", "unknown"
    filename = parts[-1]
    stem = filename.rsplit(".", 1)[0] if "." in filename else filename
    frame_id = stem or filename
    if len(parts) < 2:
        return "unknown", frame_id
    return parts[0], frame_id


def download_frame(client: Minio, frame: FrameRef) -> bytes:
    response = client.get_object(frame.bucket, frame.key)
    try:
        return response.read()
    finally:
        response.close()
        response.release_conn()


def _minio_client(settings: Settings) -> Minio:
    return Minio(
        settings.MINIO_ENDPOINT,
        access_key=settings.MINIO_ACCESS_KEY,
        secret_key=settings.MINIO_SECRET_KEY,
        secure=settings.MINIO_SECURE,
    )


async def async_sleep_while(should_continue: Callable[[], bool], seconds: float) -> bool:
    deadline = time.monotonic() + seconds
    while should_continue() and time.monotonic() < deadline:
        await asyncio.sleep(0.2)
    return should_continue()


def _class_text(prediction: FramePrediction) -> str:
    return ", ".join(f"{item.name} {item.confidence:.3f}" for item in prediction.classes)


def _object_text(prediction: FramePrediction) -> str:
    parts: list[str] = []
    for item in prediction.objects:
        box = item.bbox
        parts.append(
            f"{item.name} {item.confidence:.3f} "
            f"bbox {box.x:.3f},{box.y:.3f},{box.width:.3f},{box.height:.3f}"
        )
    return "; ".join(parts)


def prepare_frame(frame: FrameRef, client: Minio, classifier: Classifier) -> PreparedFrame | None:
    """Returns None when the object is not a frame the model can classify."""
    if not _is_image(frame.key):
        logger.info("skipping %s/%s: not jpg, jpeg, or png", frame.bucket, frame.key)
        return None
    try:
        body = download_frame(client, frame)
    except S3Error as exc:
        logger.error("failed to download %s/%s: %s", frame.bucket, frame.key, exc)
        return None
    try:
        prediction = classifier.classify(body)
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        logger.error("failed to decode %s/%s: %s", frame.bucket, frame.key, exc)
        return None

    camera_id, frame_id = frame_identity(frame.key)
    logger.info(
        "frame %s/%s (%s bytes): %s (%s)",
        frame.bucket,
        frame.key,
        len(body),
        _object_text(prediction),
        _class_text(prediction),
    )
    return PreparedFrame(camera_id, frame_id, prediction.as_dict())


class DetectionsService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.client = _minio_client(self.settings)
        self.classifier: Classifier | None = None

    def load_model(self) -> bool:
        weights = Path(self.settings.MODEL_WEIGHTS)
        try:
            self.classifier = Classifier(weights)
        except FileNotFoundError:
            logger.error("weights not found: %s", weights)
            return False
        logger.info("model loaded, classes: %s", ", ".join(self.classifier.class_names))
        return True

    async def wait_for_postgres(self, should_continue: Callable[[], bool]) -> bool:
        delay = 1.0
        while should_continue():
            try:
                async with AsyncSessionLocal() as session:
                    await DetectionsRepository(session).ping()
                return True
            except SQLAlchemyError as exc:
                logger.warning("postgres unavailable: %s", exc)
                if not await async_sleep_while(should_continue, delay):
                    return False
                delay = min(delay * 2, 10.0)
        return False

    async def consume_record(
        self,
        raw: bytes | None,
        offset: int,
        should_continue: Callable[[], bool],
    ) -> bool | None:
        """None: nothing to store, the offset can be committed. False: stopped before the row was stored."""
        refs = frame_refs(raw)
        if not refs:
            logger.warning("message has no bucket and object.key, offset=%s", offset)
            return None
        for frame in refs:
            if not await self.process_frame(frame, should_continue):
                return False
        return True

    async def process_frame(self, frame: FrameRef, should_continue: Callable[[], bool]) -> bool:
        """Returns False if the service stopped before the row was stored."""
        if self.classifier is None:
            raise RuntimeError("model is not loaded")
        prepared = await asyncio.to_thread(prepare_frame, frame, self.client, self.classifier)
        if prepared is None:
            return True

        delay = 1.0
        while should_continue():
            try:
                async with AsyncSessionLocal() as session:
                    detection_id = await DetectionsRepository(session).insert_detection(
                        prepared.frame_id,
                        prepared.camera_id,
                        frame.key,
                        prepared.payload,
                    )
            except (SQLAlchemyError, RuntimeError) as exc:
                logger.error("failed to store %s/%s: %s", frame.bucket, frame.key, exc)
                if not await async_sleep_while(should_continue, delay):
                    return False
                delay = min(delay * 2, 30.0)
                continue
            logger.info("stored detection %s for %s/%s", detection_id, frame.bucket, frame.key)
            return True
        return False
