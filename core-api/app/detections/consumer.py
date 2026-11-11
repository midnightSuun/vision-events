"""One Kafka reader for the process. Each SSE client gets a copy of new rows."""

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from aiokafka import AIOKafkaConsumer
from fastapi import FastAPI
from pydantic import ValidationError

from app.detections.dto.detection_response_dto import DetectionResponseDto
from app.settings import get_settings

logger = logging.getLogger("core-api")

_ROW_OPS = frozenset({"c", "u"})
_HEARTBEAT_SECONDS = 15.0


class DetectionHub:
    def __init__(self) -> None:
        self._queues: set[asyncio.Queue[DetectionResponseDto]] = set()

    def subscribe(self) -> asyncio.Queue[DetectionResponseDto]:
        queue: asyncio.Queue[DetectionResponseDto] = asyncio.Queue()
        self._queues.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[DetectionResponseDto]) -> None:
        self._queues.discard(queue)

    def publish(self, detection: DetectionResponseDto) -> None:
        for queue in tuple(self._queues):
            queue.put_nowait(detection)


hub = DetectionHub()


def detection_from_message(raw: bytes | None) -> DetectionResponseDto | None:
    """The Debezium note carries the new row in `after` when a row is created or updated."""
    if not raw:
        return None
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError):
        logger.warning("detection message is not json")
        return None
    if not isinstance(payload, dict) or payload.get("op") not in _ROW_OPS:
        return None
    after = payload.get("after")
    if not isinstance(after, dict):
        return None
    try:
        return DetectionResponseDto.model_validate(
            {
                "id": after.get("id"),
                "frame_id": after.get("frame_id"),
                "camera_id": after.get("camera_id"),
                "object_key": after.get("object_key"),
                "detections": _detections_payload(after.get("detections")),
                "created_at": _created_at(after.get("created_at")),
            }
        )
    except ValidationError:
        logger.warning("detection message does not match a detections row")
        return None


def _detections_payload(value: object) -> object:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _created_at(value: object) -> object:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return value
    # Debezium sends epoch microseconds for some timestamp modes, milliseconds for others.
    seconds = value / 1_000_000 if abs(value) >= 100_000_000_000_000 else value / 1_000
    return datetime.fromtimestamp(seconds, tz=timezone.utc)


async def stream_detections(queue: asyncio.Queue[DetectionResponseDto]) -> AsyncIterator[str]:
    try:
        yield ": connected\n\n"
        while True:
            try:
                detection = await asyncio.wait_for(queue.get(), timeout=_HEARTBEAT_SECONDS)
            except TimeoutError:
                yield ": ping\n\n"
                continue
            yield f"data: {detection.model_dump_json()}\n\n"
    finally:
        hub.unsubscribe(queue)


async def _consume(stop: asyncio.Event) -> None:
    settings = get_settings()
    consumer = AIOKafkaConsumer(
        settings.KAFKA_TOPIC,
        bootstrap_servers=settings.KAFKA_BOOTSTRAP_SERVERS,
        group_id=settings.KAFKA_GROUP_ID,
        client_id=settings.KAFKA_GROUP_ID,
        auto_offset_reset="latest",
        enable_auto_commit=False,
    )
    await consumer.start()
    try:
        logger.info(
            "group %s reads topic %s (%s)",
            settings.KAFKA_GROUP_ID,
            settings.KAFKA_TOPIC,
            settings.KAFKA_BOOTSTRAP_SERVERS,
        )
        while not stop.is_set():
            batches = await consumer.getmany(timeout_ms=1000, max_records=1)
            for messages in batches.values():
                for message in messages:
                    detection = detection_from_message(message.value)
                    if detection is not None:
                        hub.publish(detection)
                        logger.info(
                            "published detection %s for %s",
                            detection.id,
                            detection.object_key,
                        )
                    await consumer.commit()
    finally:
        await consumer.stop()


async def _sleep_until_stop(stop: asyncio.Event, seconds: float) -> bool:
    try:
        await asyncio.wait_for(stop.wait(), timeout=seconds)
    except TimeoutError:
        return True
    return False


async def run_consumer(stop: asyncio.Event) -> None:
    delay = 1.0
    while not stop.is_set():
        try:
            await _consume(stop)
            return
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("detection consumer failed")
            if not await _sleep_until_stop(stop, delay):
                return
            delay = min(delay * 2, 30.0)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    stop = asyncio.Event()
    task = asyncio.create_task(run_consumer(stop))
    try:
        yield
    finally:
        stop.set()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
