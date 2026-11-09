"""MinIO notification and the frame bytes the model can classify."""

import json
import logging
from typing import NamedTuple
from urllib.parse import unquote_plus

from minio import Minio
from minio.error import S3Error
from PIL import UnidentifiedImageError

from infer import Classifier, FramePrediction

logger = logging.getLogger("detection-service")


class FrameRef(NamedTuple):
    bucket: str
    key: str


class PreparedFrame(NamedTuple):
    camera_id: str
    frame_id: str
    payload: dict[str, object]


_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")


def _is_image(key: str) -> bool:
    return key.lower().endswith(_IMAGE_SUFFIXES)


def frame_refs(raw: bytes | None) -> list[FrameRef]:
    """Reads bucket and object.key from a MinIO notification."""
    if not raw:
        return []
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError):
        logger.warning("message is not json, skipping")
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
    return FrameRef(unquote_plus(bucket), unquote_plus(key))


def _key_field_refs(raw_key: object) -> list[FrameRef]:
    """Fallback MinIO format: Key = bucket/object-key."""
    if not isinstance(raw_key, str) or "/" not in raw_key:
        return []
    bucket, key = raw_key.split("/", 1)
    bucket = unquote_plus(bucket)
    key = unquote_plus(key)
    if not bucket or not key:
        return []
    return [FrameRef(bucket, key)]


def frame_identity(key: str) -> tuple[str, str]:
    """The first directory of the key is camera_id; the filename without its extension is frame_id.

    cam-1/frame-9.jpg → camera cam-1, frame frame-9.
    frame-9.jpg → camera unknown, frame frame-9.
    """
    parts = [part for part in key.split("/") if part]
    if not parts:
        return "unknown", "unknown"
    filename = parts[-1]
    stem = filename.rsplit(".", 1)[0] if "." in filename else filename
    frame_id = stem or filename
    if len(parts) < 2:
        return "unknown", frame_id
    return parts[0], frame_id


def download_frame(client: Minio, frame: FrameRef) -> bytes:
    response = client.get_object(frame.bucket, frame.key)
    try:
        return response.read()
    finally:
        response.close()
        response.release_conn()


def _class_text(prediction: FramePrediction) -> str:
    return ", ".join(f"{item.name} {item.confidence:.3f}" for item in prediction.classes)


def _object_text(prediction: FramePrediction) -> str:
    parts: list[str] = []
    for item in prediction.objects:
        box = item.bbox
        parts.append(
            f"{item.name} {item.confidence:.3f} "
            f"bbox {box.x:.3f},{box.y:.3f},{box.width:.3f},{box.height:.3f}"
        )
    return "; ".join(parts)


def prepare_frame(frame: FrameRef, client: Minio, classifier: Classifier) -> PreparedFrame | None:
    """Returns None when the object is not a frame the model can classify."""
    if not _is_image(frame.key):
        logger.info("skipping %s/%s: not jpg, jpeg, or png", frame.bucket, frame.key)
        return None
    try:
        body = download_frame(client, frame)
    except S3Error as exc:
        logger.error("failed to download %s/%s: %s", frame.bucket, frame.key, exc)
        return None
    try:
        prediction = classifier.classify(body)
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        logger.error("failed to decode %s/%s: %s", frame.bucket, frame.key, exc)
        return None

    camera_id, frame_id = frame_identity(frame.key)
    logger.info(
        "frame %s/%s (%s bytes): %s (%s)",
        frame.bucket,
        frame.key,
        len(body),
        _object_text(prediction),
        _class_text(prediction),
    )
    return PreparedFrame(camera_id, frame_id, prediction.as_dict())
