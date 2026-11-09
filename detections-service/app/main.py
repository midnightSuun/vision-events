import asyncio
import logging
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.detections.consumer import run_consumer

logger = logging.getLogger("detections-service")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    stop = threading.Event()
    consumer = asyncio.create_task(run_consumer(stop))

    def handle_consumer_done(task: asyncio.Task[None]) -> None:
        if task.cancelled():
            return
        error = task.exception()
        if error is not None:
            logger.error("consumer exited", exc_info=error)

    consumer.add_done_callback(handle_consumer_done)
    try:
        yield
    finally:
        stop.set()
        await asyncio.wait({consumer})


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
