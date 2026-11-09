"""Watch MediaMTX and store one sampled JPEG per readable camera path."""

import asyncio
import logging
import signal
import time
from urllib.parse import quote

import httpx
from minio import Minio

from app.jpegs import JpegSplitter
from app.paths import classify_page
from app.settings import Settings
from app.uploader import FrameUploader

logger = logging.getLogger("frame-grabber")

_READ_SIZE = 64 * 1024
_STOP_TIMEOUT_SECONDS = 5.0
_PAGE_SIZE = 100
_MAX_PAGES = 100


def rtsp_url(base: str, camera_id: str) -> str:
    return f"{base.rstrip('/')}/{quote(camera_id, safe='')}"


def ffmpeg_command(url: str, fps: float, quality: int) -> list[str]:
    return [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-timeout",
        "10000000",
        "-rtsp_transport",
        "tcp",
        "-i",
        url,
        "-an",
        "-vf",
        f"fps={fps:g}",
        "-q:v",
        str(quality),
        "-f",
        "image2pipe",
        "-vcodec",
        "mjpeg",
        "pipe:1",
    ]


async def run(settings: Settings) -> None:
    stop = asyncio.Event()
    _install_stop(stop)
    uploader = FrameUploader(_minio_client(settings), settings.MINIO_BUCKET)
    tasks: dict[str, asyncio.Task[None]] = {}
    warned: set[str] = set()
    logger.info(
        "watching %s, reading %s, writing %s at %s fps",
        settings.MTX_API_URL,
        settings.MTX_RTSP_BASE,
        settings.MINIO_BUCKET,
        settings.FRAME_FPS,
    )
    async with httpx.AsyncClient(base_url=settings.MTX_API_URL.rstrip("/"), timeout=5.0) as client:
        while not stop.is_set():
            ready = await _readable_cameras(client, warned)
            if ready is not None:
                await _sync_tasks(tasks, ready, settings, uploader, stop)
            if await _wait(stop, settings.POLL_INTERVAL_SECONDS):
                break
    for task in tasks.values():
        task.cancel()
    if tasks:
        await asyncio.gather(*tasks.values(), return_exceptions=True)


def _install_stop(stop: asyncio.Event) -> None:
    loop = asyncio.get_running_loop()

    def request_stop() -> None:
        logger.info("stopping")
        stop.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, request_stop)
        except NotImplementedError:
            signal.signal(sig, lambda *_args: request_stop())


def _minio_client(settings: Settings) -> Minio:
    return Minio(
        settings.MINIO_ENDPOINT,
        access_key=settings.MINIO_ACCESS_KEY,
        secret_key=settings.MINIO_SECRET_KEY,
        secure=settings.MINIO_SECURE,
    )


async def _readable_cameras(client: httpx.AsyncClient, warned: set[str]) -> set[str] | None:
    try:
        camera_ids: set[str] = set()
        page = 0
        while page < _MAX_PAGES:
            response = await client.get(
                "/v3/paths/list",
                params={"page": page, "itemsPerPage": _PAGE_SIZE},
            )
            response.raise_for_status()
            found, skipped, page_count = classify_page(response.json())
            for name in skipped:
                if name in warned:
                    continue
                logger.warning("skipping path %s: camera id must be one path segment", name)
                warned.add(name)
            camera_ids.update(found)
            page += 1
            if page >= page_count:
                return camera_ids
        return camera_ids
    except (httpx.HTTPError, ValueError, TypeError) as exc:
        logger.warning("mediamtx api unavailable: %s", exc)
        return None


async def _sync_tasks(
    tasks: dict[str, asyncio.Task[None]],
    ready: set[str],
    settings: Settings,
    uploader: FrameUploader,
    stop: asyncio.Event,
) -> None:
    if stop.is_set():
        return
    closing = [tasks.pop(camera_id) for camera_id in list(tasks) if camera_id not in ready]
    for task in closing:
        task.cancel()
    if closing:
        await asyncio.gather(*closing, return_exceptions=True)
    for camera_id in ready:
        current = tasks.get(camera_id)
        if current is not None and not current.done():
            continue
        if current is not None:
            _log_task_error(camera_id, current)
        tasks[camera_id] = asyncio.create_task(
            _grab_camera(camera_id, settings, uploader, stop),
            name=f"grab-{camera_id}",
        )


def _log_task_error(camera_id: str, task: asyncio.Task[None]) -> None:
    if task.cancelled():
        return
    error = task.exception()
    if error is not None:
        logger.error("grabber for %s stopped", camera_id, exc_info=error)


async def _grab_camera(
    camera_id: str,
    settings: Settings,
    uploader: FrameUploader,
    stop: asyncio.Event,
) -> None:
    delay = 1.0
    while not stop.is_set():
        stored = await _stream_once(camera_id, settings, uploader, stop)
        if stop.is_set():
            return
        if stored:
            delay = 1.0
        logger.info("stream %s ended, retrying in %.0fs", camera_id, delay)
        if await _wait(stop, delay):
            return
        delay = min(delay * 2, 15.0)


async def _stream_once(
    camera_id: str,
    settings: Settings,
    uploader: FrameUploader,
    stop: asyncio.Event,
) -> int:
    url = rtsp_url(settings.MTX_RTSP_BASE, camera_id)
    logger.info("reading %s", url)
    proc = await asyncio.create_subprocess_exec(
        *ffmpeg_command(url, settings.FRAME_FPS, settings.JPEG_QUALITY),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    if proc.stdout is None or proc.stderr is None:
        await _stop_ffmpeg(proc)
        raise RuntimeError("ffmpeg pipes were not created")
    stderr_task = asyncio.create_task(_drain_stderr(proc.stderr, camera_id))
    splitter = JpegSplitter()
    stored = 0
    min_interval = 0.9 / settings.FRAME_FPS
    last_stored_at = 0.0
    try:
        while not stop.is_set():
            chunk = await proc.stdout.read(_READ_SIZE)
            if not chunk:
                break
            for payload in splitter.feed(chunk):
                now = time.monotonic()
                if now - last_stored_at < min_interval:
                    continue
                frame_id = str(time.time_ns())
                saved = await asyncio.to_thread(uploader.put_jpeg, camera_id, frame_id, payload)
                if saved:
                    stored += 1
                    last_stored_at = now
        return stored
    finally:
        await _finish_process(proc, stderr_task)


async def _drain_stderr(stream: asyncio.StreamReader, camera_id: str) -> None:
    while True:
        line = await stream.readline()
        if not line:
            return
        text = line.decode("utf-8", errors="replace").strip()
        if text:
            logger.warning("ffmpeg %s: %s", camera_id, text)


async def _finish_process(
    proc: asyncio.subprocess.Process,
    stderr_task: asyncio.Task[None],
) -> None:
    """Stop ffmpeg even when this task is already being cancelled."""
    current = asyncio.current_task()
    cancelled = current is not None and current.cancelling() > 0
    if cancelled and current is not None:
        while current.cancelling():
            current.uncancel()
    try:
        await _stop_ffmpeg(proc)
        if not stderr_task.done():
            stderr_task.cancel()
        await asyncio.gather(stderr_task, return_exceptions=True)
    finally:
        if cancelled:
            raise asyncio.CancelledError


async def _stop_ffmpeg(proc: asyncio.subprocess.Process) -> None:
    if proc.returncode is not None:
        return
    try:
        proc.terminate()
    except ProcessLookupError:
        return
    try:
        await asyncio.wait_for(proc.wait(), _STOP_TIMEOUT_SECONDS)
    except TimeoutError:
        try:
            proc.kill()
        except ProcessLookupError:
            return
        await proc.wait()


async def _wait(stop: asyncio.Event, seconds: float) -> bool:
    try:
        await asyncio.wait_for(stop.wait(), timeout=seconds)
    except TimeoutError:
        return False
    return True
