from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    MTX_API_URL: str = "http://mediamtx:9997"
    MTX_RTSP_BASE: str = "rtsp://mediamtx:8554"
    MINIO_ENDPOINT: str = "minio:9000"
    MINIO_ACCESS_KEY: str = "minioadmin"
    MINIO_SECRET_KEY: str = "minioadmin"
    MINIO_SECURE: bool = False
    MINIO_BUCKET: str = "frames"
    FRAME_FPS: float = Field(default=1.0, gt=0)
    JPEG_QUALITY: int = Field(default=5, ge=2, le=31)
    POLL_INTERVAL_SECONDS: float = Field(default=2.0, gt=0)
    LOG_LEVEL: str = "INFO"

    model_config = SettingsConfigDict(env_file=".env", extra="allow")


@lru_cache
def get_settings() -> Settings:
    return Settings()
