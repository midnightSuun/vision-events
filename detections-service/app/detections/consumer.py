"""Kafka loop. A message becomes a frame, and DetectionService stores it.

The row is the event. Debezium reads the Postgres log and publishes it to Kafka,
so this service does not write to Kafka itself.
"""

import asyncio
import logging
import signal
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path

from aiokafka import AIOKafkaConsumer
from minio import Minio
from sqlalchemy.exc import SQLAlchemyError

from app.db import AsyncSessionLocal
from app.detections.detections_repository import DetectionsRepository
from app.detections.detections_service import DetectionService
from app.detections.frames import frame_refs
from app.settings import Settings, get_settings
from infer import Classifier

logging.basicConfig(
    level=get_settings().LOG_LEVEL,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("detection-service")


def _minio_client(settings: Settings) -> Minio:
    return Minio(
        settings.MINIO_ENDPOINT,
        access_key=settings.MINIO_ACCESS_KEY,
        secret_key=settings.MINIO_SECRET_KEY,
        secure=settings.MINIO_SECURE,
    )


async def _sleep_while(should_continue: Callable[[], bool], seconds: float) -> bool:
    deadline = time.monotonic() + seconds
    while should_continue() and time.monotonic() < deadline:
        await asyncio.sleep(0.2)
    return should_continue()


async def _wait_for_postgres(
    repository: DetectionsRepository,
    should_continue: Callable[[], bool],
) -> bool:
    delay = 1.0
    while should_continue():
        try:
            await repository.ping()
        except SQLAlchemyError as exc:
            logger.warning("postgres unavailable: %s", exc)
            if not await _sleep_while(should_continue, delay):
                return False
            delay = min(delay * 2, 10.0)
            continue
        return True
    return False


async def _consume(stop_event: threading.Event) -> None:
    settings = get_settings()

    def should_continue() -> bool:
        return not stop_event.is_set()

    try:
        classifier = Classifier(Path(settings.MODEL_WEIGHTS))
    except FileNotFoundError:
        logger.error("weights not found: %s", settings.MODEL_WEIGHTS)
        return
    logger.info("model loaded, classes: %s", ", ".join(classifier.class_names))

    repository = DetectionsRepository(AsyncSessionLocal)
    service = DetectionService(repository, _minio_client(settings), classifier)
    if not await _wait_for_postgres(repository, should_continue):
        return

    consumer = AIOKafkaConsumer(
        settings.KAFKA_TOPIC,
        bootstrap_servers=settings.KAFKA_BOOTSTRAP_SERVERS,
        group_id=settings.KAFKA_GROUP_ID,
        client_id=settings.KAFKA_GROUP_ID,
        auto_offset_reset="earliest",
        enable_auto_commit=False,
    )
    await consumer.start()
    try:
        logger.info(
            "group %s reads topic %s (%s), frames from %s, rows go to postgres",
            settings.KAFKA_GROUP_ID,
            settings.KAFKA_TOPIC,
            settings.KAFKA_BOOTSTRAP_SERVERS,
            settings.MINIO_ENDPOINT,
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
                        if not await service.process_frame(frame, should_continue):
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
            if not await _sleep_while(lambda: not stop_event.is_set(), delay):
                return
            delay = min(delay * 2, 30.0)


def main() -> None:
    asyncio.run(run_consumer())


if __name__ == "__main__":
    main()
