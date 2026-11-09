from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.models import Detections


class DetectionsRepository:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]):
        self.session_factory = session_factory

    async def ping(self) -> None:
        async with self.session_factory() as db:
            await db.execute(text("SELECT 1"))

    async def insert_detection(
        self,
        frame_id: str,
        camera_id: str,
        object_key: str,
        detections: dict[str, object],
    ) -> int:
        async with self.session_factory() as db:
            row = Detections(
                frame_id=frame_id,
                camera_id=camera_id,
                object_key=object_key,
                detections=detections,
            )
            db.add(row)
            await db.flush()
            detection_id = row.id
            if detection_id is None:
                raise RuntimeError("postgres did not return a detection id")
            await db.commit()
            return int(detection_id)
