from functools import lru_cache
from urllib.parse import urlparse, urlunparse

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    DATABASE_URL: str = "postgresql://vision:vision@postgres:5432/vision"
    KAFKA_BOOTSTRAP_SERVERS: str = "kafka:19092"
    KAFKA_TOPIC: str = "frame_created"
    KAFKA_GROUP_ID: str = "detection-service"
    MINIO_ENDPOINT: str = "minio:9000"
    MINIO_ACCESS_KEY: str = "minioadmin"
    MINIO_SECRET_KEY: str = "minioadmin"
    MINIO_SECURE: bool = False
    MODEL_WEIGHTS: str = "/app/ml/artifacts/model.npz"
    LOG_LEVEL: str = "INFO"

    model_config = SettingsConfigDict(env_file=".env", extra="allow")

    @property
    def sqlalchemy_url(self) -> str:
        parsed = urlparse(self.DATABASE_URL)
        return urlunparse(parsed._replace(scheme="postgresql+asyncpg"))


@lru_cache
def get_settings() -> Settings:
    return Settings()
