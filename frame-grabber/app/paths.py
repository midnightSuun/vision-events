"""MediaMTX path list and the camera id stored in the object key."""

from typing import NamedTuple


class PathPage(NamedTuple):
    camera_ids: set[str]
    skipped: set[str]
    page_count: int


def camera_id_from_path(path: str) -> str | None:
    """The object key is {camera_id}/{frame_id}.jpg, so the path must be one segment.

    detections-service reads the camera id from the first directory of the key.
    """
    name = path.strip("/")
    if not name or "/" in name or name in {".", ".."}:
        return None
    return name


def classify_page(payload: object) -> PathPage:
    if not isinstance(payload, dict):
        raise TypeError("paths list is not an object")
    items = payload.get("items") or []
    if not isinstance(items, list):
        raise TypeError("paths list items is not a list")

    camera_ids: set[str] = set()
    skipped: set[str] = set()
    for item in items:
        if not isinstance(item, dict) or not _is_readable(item):
            continue
        name = item.get("name")
        if not isinstance(name, str) or not name:
            continue
        camera_id = camera_id_from_path(name)
        if camera_id is None:
            skipped.add(name)
            continue
        camera_ids.add(camera_id)

    page_count = payload.get("pageCount") or 0
    if not isinstance(page_count, int):
        page_count = 0
    return PathPage(camera_ids, skipped, page_count)


def _is_readable(item: dict[str, object]) -> bool:
    if "available" in item:
        return bool(item["available"])
    return bool(item.get("ready"))
