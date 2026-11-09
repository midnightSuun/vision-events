import unittest

from app.paths import camera_id_from_path, classify_page


class PathTest(unittest.TestCase):
    def test_camera_id_is_a_single_segment(self) -> None:
        self.assertEqual(camera_id_from_path("cam-1"), "cam-1")
        self.assertEqual(camera_id_from_path("/cam-1/"), "cam-1")
        self.assertIsNone(camera_id_from_path("floor/cam-1"))
        self.assertIsNone(camera_id_from_path(""))

    def test_readable_paths_prefer_available(self) -> None:
        page = classify_page(
            {
                "pageCount": 1,
                "items": [
                    {"name": "cam-1", "available": True, "ready": False},
                    {"name": "cam-2", "available": False, "ready": True},
                    {"name": "floor/cam-3", "available": True},
                ],
            }
        )
        self.assertEqual(page.camera_ids, {"cam-1"})
        self.assertEqual(page.skipped, {"floor/cam-3"})
        self.assertEqual(page.page_count, 1)

    def test_ready_is_used_when_available_is_absent(self) -> None:
        page = classify_page({"items": [{"name": "cam-1", "ready": True}]})
        self.assertEqual(page.camera_ids, {"cam-1"})


if __name__ == "__main__":
    unittest.main()
