"""detection-service: читает frame_created и скачивает кадр из MinIO."""

from __future__ import annotations

import json
import logging
import os
import signal
import sys
from typing import NamedTuple
from urllib.parse import unquote

from confluent_kafka import Consumer, KafkaError, KafkaException
from minio import Minio
from minio.error import S3Error

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


def frame_refs(raw: bytes | None) -> list[FrameRef]:
    """Достаёт bucket и object.key из уведомления MinIO."""
    if not raw:
        return []
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError):
        logger.warning("сообщение не json, пропуск")
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
    return FrameRef(unquote(bucket), unquote(key))


def _key_field_refs(raw_key: object) -> list[FrameRef]:
    """Запасной формат MinIO: Key = bucket/object-key."""
    if not isinstance(raw_key, str) or "/" not in raw_key:
        return []
    bucket, key = raw_key.split("/", 1)
    bucket = unquote(bucket)
    key = unquote(key)
    if not bucket or not key:
        return []
    return [FrameRef(bucket, key)]


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


def main() -> None:
    bootstrap = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "kafka:19092")
    topic = os.environ.get("KAFKA_TOPIC", "frame_created")
    group_id = os.environ.get("KAFKA_GROUP_ID", "detection-service")
    endpoint = os.environ.get("MINIO_ENDPOINT", "minio:9000")
    client = minio_client()

    consumer = Consumer(
        {
            "bootstrap.servers": bootstrap,
            "group.id": group_id,
            "client.id": group_id,
            "auto.offset.reset": "earliest",
            "enable.auto.commit": True,
        }
    )
    consumer.subscribe([topic])

    running = True

    def handle_stop(signum: int, _frame: object) -> None:
        nonlocal running
        running = False
        logger.info("остановка по сигналу %s", signum)

    signal.signal(signal.SIGTERM, handle_stop)
    signal.signal(signal.SIGINT, handle_stop)

    logger.info(
        "группа %s читает топик %s (%s), кадры из %s",
        group_id,
        topic,
        bootstrap,
        endpoint,
    )
    try:
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
                logger.warning("в сообщении нет bucket и object.key, offset=%s", message.offset())
                continue
            for frame in refs:
                try:
                    body = download_frame(client, frame)
                except S3Error as exc:
                    logger.error(
                        "не удалось скачать %s/%s: %s",
                        frame.bucket,
                        frame.key,
                        exc,
                    )
                    continue
                logger.info(
                    "скачан кадр %s/%s, %s байт",
                    frame.bucket,
                    frame.key,
                    len(body),
                )
    finally:
        consumer.close()


if __name__ == "__main__":
    main()
