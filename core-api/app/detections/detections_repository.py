from sqlalchemy import select

from app.db import DbSession
from app.models import Detections


class DetectionsRepository:
    def __init__(self, db: DbSession):
        self.db = db

    async def get_detections(self, limit: int, camera_id: str | None) -> list[Detections]:
        statement = select(Detections).order_by(Detections.created_at.desc()).limit(limit)
        if camera_id is not None:
            statement = statement.where(Detections.camera_id == camera_id)
        result = await self.db.execute(statement)
        return list(result.scalars().all())
