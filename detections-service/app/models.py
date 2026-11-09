"""Generated from the live Postgres schema by sqlacodegen. Do not edit by hand.

Regenerate after applying a dbmate migration:

    uv run python -m sqlacodegen "postgresql+psycopg2://vision:vision@localhost:5432/vision" --outfile app/models.py
"""

import datetime

from sqlalchemy import (
    BigInteger,
    DateTime,
    Index,
    PrimaryKeyConstraint,
    String,
    Text,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Detections(Base):
    __tablename__ = 'detections'
    __table_args__ = (
        PrimaryKeyConstraint('id', name='detections_pkey'),
        Index('detections_camera_created_at_idx', 'camera_id', 'created_at')
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    frame_id: Mapped[str] = mapped_column(Text, nullable=False)
    camera_id: Mapped[str] = mapped_column(Text, nullable=False)
    object_key: Mapped[str] = mapped_column(Text, nullable=False)
    detections: Mapped[object] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime(True), nullable=False, server_default=text('now()'))


class SchemaMigrations(Base):
    __tablename__ = 'schema_migrations'
    __table_args__ = (
        PrimaryKeyConstraint('version', name='schema_migrations_pkey'),
    )

    version: Mapped[str] = mapped_column(String, primary_key=True)
