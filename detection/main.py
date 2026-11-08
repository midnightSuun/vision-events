"""detection-service: frame from MinIO → model class and box → Postgres and detection_completed."""

from __future__ import annotations

import json
import logging
import os
import signal
import sys
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import NamedTuple
from urllib.parse import unquote_plus

import psycopg
from confluent_kafka import Consumer, KafkaError, KafkaException, Producer
from confluent_kafka.admin import AdminClient, NewTopic
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


def ensure_topic(bootstrap: str, topic: str) -> None:
    admin = AdminClient({"bootstrap.servers": bootstrap})
    future = admin.create_topics(
        [NewTopic(topic, num_partitions=1, replication_factor=1)],
        request_timeout=30,
    )[topic]
    try:
        future.result()
    except KafkaException as exc:
        error = exc.args[0] if exc.args else None
        code = error.code() if isinstance(error, KafkaError) else None
        if code == KafkaError.TOPIC_ALREADY_EXISTS:
            return
        raise
    logger.info("created topic %s", topic)


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


def publish_detection(producer: Producer, topic: str, event: dict[str, object]) -> None:
    errors: list[str] = []

    def on_delivery(err: KafkaError | None, _msg: object) -> None:
        if err is not None:
            errors.append(str(err))

    producer.produce(
        topic,
        key=str(event["camera_id"]).encode("utf-8"),
        value=json.dumps(event, ensure_ascii=False).encode("utf-8"),
        on_delivery=on_delivery,
    )
    leftover = producer.flush(10.0)
    if leftover or errors:
        detail = errors[0] if errors else f"{leftover} messages left in the queue"
        raise RuntimeError(detail)


def sleep_while(should_continue: Callable[[], bool], seconds: float) -> bool:
    deadline = time.monotonic() + seconds
    while should_continue() and time.monotonic() < deadline:
        time.sleep(0.2)
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


def process_frame(
    frame: FrameRef,
    client: Minio,
    classifier: Classifier,
    producer: Producer,
    database_url: str,
    detection_topic: str,
    should_continue: Callable[[], bool],
) -> bool:
    """Returns False if the service stopped before the write, so the publish cannot be confirmed."""
    if not _is_image(frame.key):
        logger.info("skipping %s/%s: not jpg, jpeg, or png", frame.bucket, frame.key)
        return True
    try:
        body = download_frame(client, frame)
    except S3Error as exc:
        logger.error("failed to download %s/%s: %s", frame.bucket, frame.key, exc)
        return True
    try:
        prediction = classifier.classify(body)
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        logger.error("failed to decode %s/%s: %s", frame.bucket, frame.key, exc)
        return True

    camera_id, frame_id = frame_identity(frame.key)
    payload = prediction.as_dict()
    logger.info(
        "frame %s/%s (%s bytes): %s (%s)",
        frame.bucket,
        frame.key,
        len(body),
        _object_text(prediction),
        _class_text(prediction),
    )

    saved: tuple[int, datetime] | None = None
    delay = 1.0
    while should_continue():
        try:
            if saved is None:
                saved = insert_detection(database_url, frame_id, camera_id, frame.key, payload)
            detection_id, created_at = saved
            publish_detection(
                producer,
                detection_topic,
                {
                    "id": detection_id,
                    "frame_id": frame_id,
                    "camera_id": camera_id,
                    "object_key": frame.key,
                    "detections": payload,
                    "created_at": created_at.isoformat(),
                },
            )
        except (psycopg.Error, RuntimeError, KafkaException, BufferError) as exc:
            logger.error("failed to store %s/%s: %s", frame.bucket, frame.key, exc)
            if not sleep_while(should_continue, delay):
                return False
            delay = min(delay * 2, 30.0)
            continue
        logger.info(
            "detection %s for %s/%s published to %s",
            detection_id,
            frame.bucket,
            frame.key,
            detection_topic,
        )
        return True
    return False


def main() -> None:
    bootstrap = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "kafka:19092")
    topic = os.environ.get("KAFKA_TOPIC", "frame_created")
    detection_topic = os.environ.get("KAFKA_DETECTION_TOPIC", "detection_completed")
    group_id = os.environ.get("KAFKA_GROUP_ID", "detection-service")
    endpoint = os.environ.get("MINIO_ENDPOINT", "minio:9000")
    database_url = os.environ.get(
        "DATABASE_URL",
        "postgresql://vision:vision@postgres:5432/vision",
    )
    weights = Path(os.environ.get("MODEL_WEIGHTS", "/app/ml/artifacts/model.npz"))
    client = minio_client()

    running = True

    def should_continue() -> bool:
        return running

    def handle_stop(signum: int, _frame: object) -> None:
        nonlocal running
        running = False
        logger.info("stopping on signal %s", signum)

    signal.signal(signal.SIGTERM, handle_stop)
    signal.signal(signal.SIGINT, handle_stop)

    try:
        classifier = Classifier(weights)
    except FileNotFoundError:
        logger.error("weights not found: %s", weights)
        return
    logger.info("model loaded, classes: %s", ", ".join(classifier.class_names))

    producer = Producer(
        {
            "bootstrap.servers": bootstrap,
            "client.id": f"{group_id}-publisher",
            "acks": "all",
            "message.timeout.ms": 10000,
        }
    )
    consumer: Consumer | None = None
    try:
        ensure_topic(bootstrap, detection_topic)
        if not wait_for_postgres(database_url, should_continue):
            return
        consumer = Consumer(
            {
                "bootstrap.servers": bootstrap,
                "group.id": group_id,
                "client.id": group_id,
                "auto.offset.reset": "earliest",
                "enable.auto.commit": False,
            }
        )
        consumer.subscribe([topic])
        logger.info(
            "group %s reads topic %s (%s), frames from %s, result to %s",
            group_id,
            topic,
            bootstrap,
            endpoint,
            detection_topic,
        )
        while running:
            message = consumer.poll(1.0)
            if message is None:
                continue
            error = message.error()
            if error is not None:
                if error.code() == KafkaError._PARTITION_EOF:
                    continue
                raise KafkaException(error)
            refs = frame_refs(message.value())
            if not refs:
                logger.warning("message has no bucket and object.key, offset=%s", message.offset())
                consumer.commit(message=message)
                continue
            stored = True
            for frame in refs:
                if not process_frame(
                    frame,
                    client,
                    classifier,
                    producer,
                    database_url,
                    detection_topic,
                    should_continue,
                ):
                    stored = False
                    break
            if not stored:
                break
            consumer.commit(message=message)
    finally:
        if consumer is not None:
            consumer.close()
        producer.flush(5)


if __name__ == "__main__":
    main()
