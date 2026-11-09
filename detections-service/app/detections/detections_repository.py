from sqlalchemy import text

from app.db import DbSession
from app.models import Detections


class DetectionsRepository:
    def __init__(self, db: DbSession):
        self.db = db

    async def ping(self) -> None:
        await self.db.execute(text("SELECT 1"))

    async def insert_detection(
        self,
        frame_id: str,
        camera_id: str,
        object_key: str,
        detections: dict[str, object],
    ) -> int:
        row = Detections(
            frame_id=frame_id,
            camera_id=camera_id,
            object_key=object_key,
            detections=detections,
        )
        self.db.add(row)
        await self.db.flush()
        detection_id = row.id
        if detection_id is None:
            raise RuntimeError("postgres did not return a detection id")
        await self.db.commit()
        return int(detection_id)
