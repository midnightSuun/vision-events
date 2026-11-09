"""detection-service: frame from MinIO → model class and box → Postgres.

The row is the event. Debezium reads the Postgres log and publishes it to Kafka,
so this service does not write to Kafka itself.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import sys
import threading
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import NamedTuple
from urllib.parse import unquote_plus

import psycopg
from aiokafka import AIOKafkaConsumer
from minio import Minio
from minio.error import S3Error
from PIL import UnidentifiedImageError
from psycopg.types.json import Json

from infer import Classifier, FramePrediction

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    stream=sys.stdout,
)
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


def minio_client() -> Minio:
    endpoint = os.environ.get("MINIO_ENDPOINT", "minio:9000")
    secure = os.environ.get("MINIO_SECURE", "false").lower() == "true"
    return Minio(
        endpoint,
        access_key=os.environ.get("MINIO_ACCESS_KEY", "minioadmin"),
        secret_key=os.environ.get("MINIO_SECRET_KEY", "minioadmin"),
        secure=secure,
    )


def wait_for_postgres(database_url: str, should_continue: Callable[[], bool]) -> bool:
    delay = 1.0
    while should_continue():
        try:
            with psycopg.connect(database_url, connect_timeout=5) as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
            return True
        except psycopg.Error as exc:
            logger.warning("postgres unavailable: %s", exc)
            if not sleep_while(should_continue, delay):
                return False
            delay = min(delay * 2, 10.0)
    return False


def insert_detection(
    database_url: str,
    frame_id: str,
    camera_id: str,
    object_key: str,
    detections: dict[str, object],
) -> tuple[int, datetime]:
    with psycopg.connect(database_url, connect_timeout=5) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO detections (frame_id, camera_id, object_key, detections)
                VALUES (%s, %s, %s, %s)
                RETURNING id, created_at
                """,
                (frame_id, camera_id, object_key, Json(detections)),
            )
            row = cur.fetchone()
        if row is None:
            raise RuntimeError("postgres did not return a detection id")
        return int(row[0]), row[1]


def sleep_while(should_continue: Callable[[], bool], seconds: float) -> bool:
    deadline = time.monotonic() + seconds
    while should_continue() and time.monotonic() < deadline:
        time.sleep(0.2)
    return should_continue()


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


async def process_frame(
    frame: FrameRef,
    client: Minio,
    classifier: Classifier,
    database_url: str,
    should_continue: Callable[[], bool],
) -> bool:
    """Returns False if the service stopped before the row was stored."""
    prepared = await asyncio.to_thread(prepare_frame, frame, client, classifier)
    if prepared is None:
        return True

    delay = 1.0
    while should_continue():
        try:
            detection_id, _created_at = await asyncio.to_thread(
                insert_detection,
                database_url,
                prepared.frame_id,
                prepared.camera_id,
                frame.key,
                prepared.payload,
            )
        except (psycopg.Error, RuntimeError) as exc:
            logger.error("failed to store %s/%s: %s", frame.bucket, frame.key, exc)
            if not await async_sleep_while(should_continue, delay):
                return False
            delay = min(delay * 2, 30.0)
            continue
        logger.info("stored detection %s for %s/%s", detection_id, frame.bucket, frame.key)
        return True
    return False


async def _consume(stop_event: threading.Event) -> None:
    bootstrap = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "kafka:19092")
    topic = os.environ.get("KAFKA_TOPIC", "frame_created")
    group_id = os.environ.get("KAFKA_GROUP_ID", "detection-service")
    endpoint = os.environ.get("MINIO_ENDPOINT", "minio:9000")
    database_url = os.environ.get(
        "DATABASE_URL",
        "postgresql://vision:vision@postgres:5432/vision",
    )
    weights = Path(os.environ.get("MODEL_WEIGHTS", "/app/ml/artifacts/model.npz"))
    client = minio_client()

    def should_continue() -> bool:
        return not stop_event.is_set()

    try:
        classifier = Classifier(weights)
    except FileNotFoundError:
        logger.error("weights not found: %s", weights)
        return
    logger.info("model loaded, classes: %s", ", ".join(classifier.class_names))

    if not await asyncio.to_thread(wait_for_postgres, database_url, should_continue):
        return

    consumer = AIOKafkaConsumer(
        topic,
        bootstrap_servers=bootstrap,
        group_id=group_id,
        client_id=group_id,
        auto_offset_reset="earliest",
        enable_auto_commit=False,
    )
    await consumer.start()
    try:
        logger.info(
            "group %s reads topic %s (%s), frames from %s, rows go to postgres",
            group_id,
            topic,
            bootstrap,
            endpoint,
        )
        while should_continue():
            batches = await consumer.getmany(timeout_ms=1000, max_records=1)
            if not batches:
                continue
            for messages in batches.values():
                for message in messages:
                    refs = frame_refs(message.value)
                    if not refs:
                        logger.warning(
                            "message has no bucket and object.key, offset=%s",
                            message.offset,
                        )
                        await consumer.commit()
                        continue
                    stored = True
                    for frame in refs:
                        if not await process_frame(
                            frame,
                            client,
                            classifier,
                            database_url,
                            should_continue,
                        ):
                            stored = False
                            break
                    if not stored:
                        return
                    await consumer.commit()
    finally:
        await consumer.stop()


async def run_consumer(stop: threading.Event | None = None) -> None:
    stop_event = threading.Event() if stop is None else stop
    if stop is None:

        def handle_stop(signum: int, _frame: object) -> None:
            stop_event.set()
            logger.info("stopping on signal %s", signum)

        signal.signal(signal.SIGTERM, handle_stop)
        signal.signal(signal.SIGINT, handle_stop)

    delay = 1.0
    while not stop_event.is_set():
        try:
            await _consume(stop_event)
            return
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("consumer failed")
            if stop_event.is_set():
                return
            if not await async_sleep_while(lambda: not stop_event.is_set(), delay):
                return
            delay = min(delay * 2, 30.0)


def main() -> None:
    asyncio.run(run_consumer())


if __name__ == "__main__":
    main()
