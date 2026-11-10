from fastapi import FastAPI

from app.detections import router as detections_router

app = FastAPI()

app.include_router(detections_router)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
