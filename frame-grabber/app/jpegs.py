"""Split a concatenated MJPEG byte stream into whole JPEG frames."""

import logging

logger = logging.getLogger("frame-grabber")

_SOI = b"\xff\xd8"
_EOI = b"\xff\xd9"
_MAX_BUFFER = 8 * 1024 * 1024


class JpegSplitter:
    def __init__(self) -> None:
        self._buffer = bytearray()

    def feed(self, chunk: bytes) -> list[bytes]:
        self._buffer.extend(chunk)
        frames: list[bytes] = []
        while True:
            start = self._buffer.find(_SOI)
            if start < 0:
                if len(self._buffer) > 1:
                    del self._buffer[:-1]
                break
            if start:
                del self._buffer[:start]
            end = self._buffer.find(_EOI, len(_SOI))
            if end < 0:
                break
            frames.append(bytes(self._buffer[: end + len(_EOI)]))
            del self._buffer[: end + len(_EOI)]
        if len(self._buffer) > _MAX_BUFFER:
            logger.warning("jpeg buffer exceeded %s bytes, dropping a partial frame", _MAX_BUFFER)
            self._buffer.clear()
        return frames
