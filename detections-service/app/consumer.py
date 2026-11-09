"""Kafka loop. Each message is handled by DetectionsService.

The row is the event. Debezium reads the Postgres log and publishes it to Kafka,
so this service does not write to Kafka itself.
"""

from __future__ import annotations

import asyncio
import logging
import signal
import sys
import threading

from aiokafka import AIOKafkaConsumer

from app.detections.detections_service import DetectionsService, async_sleep_while
from app.settings import get_settings

logging.basicConfig(
    level=get_settings().LOG_LEVEL,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("detection-service")


async def _consume(stop_event: threading.Event) -> None:
    settings = get_settings()
    service = DetectionsService()

    def should_continue() -> bool:
        return not stop_event.is_set()

    if not service.load_model():
        return
    if not await service.wait_for_postgres(should_continue):
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
                    stored = await service.consume_record(
                        message.value,
                        message.offset,
                        should_continue,
                    )
                    if stored is None:
                        await consumer.commit()
                        continue
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
