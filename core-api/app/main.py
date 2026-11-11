from fastapi import FastAPI

from app.detections import router as detections_router
from app.detections.consumer import lifespan

app = FastAPI(lifespan=lifespan)

app.include_router(detections_router)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
