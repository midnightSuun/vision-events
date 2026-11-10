import unittest

from app.jpegs import JpegSplitter


def _jpeg(body: bytes) -> bytes:
    return b"\xff\xd8" + body + b"\xff\xd9"


class JpegSplitterTest(unittest.TestCase):
    def test_two_frames_in_one_chunk(self) -> None:
        first = _jpeg(b"one")
        second = _jpeg(b"two")
        self.assertEqual(JpegSplitter().feed(first + second), [first, second])

    def test_frame_split_across_chunks(self) -> None:
        frame = _jpeg(b"abcdef")
        splitter = JpegSplitter()
        self.assertEqual(splitter.feed(frame[:3]), [])
        self.assertEqual(splitter.feed(frame[3:]), [frame])

    def test_marker_split_across_chunks(self) -> None:
        frame = _jpeg(b"ab")
        splitter = JpegSplitter()
        self.assertEqual(splitter.feed(frame[:1]), [])
        self.assertEqual(splitter.feed(frame[1:]), [frame])

    def test_leading_garbage_is_dropped(self) -> None:
        frame = _jpeg(b"ok")
        self.assertEqual(JpegSplitter().feed(b"noise" + frame), [frame])


if __name__ == "__main__":
    unittest.main()
