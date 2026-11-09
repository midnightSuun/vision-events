from collections.abc import AsyncGenerator
from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.settings import get_settings


def async_database_url(database_url: str) -> str:
    scheme, separator, rest = database_url.partition("://")
    if separator != "://":
        return database_url
    if scheme in {"postgresql", "postgres", "postgresql+asyncpg", "postgres+asyncpg"}:
        return f"postgresql+asyncpg://{rest}"
    return database_url


settings = get_settings()
engine = create_async_engine(async_database_url(settings.DATABASE_URL), echo=True)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


class Base(DeclarativeBase):
    pass


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        yield session


DbSession = Annotated[AsyncSession, Depends(get_db)]
