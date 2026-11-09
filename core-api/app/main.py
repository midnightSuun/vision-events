import asyncio
import os
from typing import Annotated

import psycopg
from fastapi import FastAPI, Query
from psycopg.rows import dict_row

from app.db_models import Detections

app = FastAPI()


def database_url() -> str:
    return os.environ.get(
        "DATABASE_URL",
        "postgresql://vision:vision@localhost:5432/vision",
    )


def fetch_detections(url: str, limit: int, camera_id: str | None) -> list[Detections]:
    sql = "SELECT * FROM detections"
    params: list[object] = []
    if camera_id is not None:
        sql += " WHERE camera_id = %s"
        params.append(camera_id)
    sql += " ORDER BY created_at DESC LIMIT %s"
    params.append(limit)
    with psycopg.connect(url, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
    return [Detections.model_validate(row) for row in rows]


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/detections")
async def list_detections(
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    camera_id: str | None = None,
) -> list[Detections]:
    return await asyncio.to_thread(fetch_detections, database_url(), limit, camera_id)
