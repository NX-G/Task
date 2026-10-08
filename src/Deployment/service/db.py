"""Postgres: job tracking, request logs and user feedback."""

import datetime as dt
import uuid
from contextlib import contextmanager
from functools import lru_cache

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from drawing_ai.config import get_settings


def _now():
    return dt.datetime.now(dt.timezone.utc)


class Base(DeclarativeBase):
    pass


class JobStatus:
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    status: Mapped[str] = mapped_column(String(16), default=JobStatus.QUEUED, index=True)
    filename: Mapped[str] = mapped_column(String(512))
    image_id: Mapped[str] = mapped_column(String(128), nullable=True)
    input_file_id: Mapped[str] = mapped_column(String(64))
    executor: Mapped[str] = mapped_column(String(16), default="inline")
    external_run_id: Mapped[str] = mapped_column(String(256), nullable=True)  # Airflow dag_run_id
    summary: Mapped[dict] = mapped_column(JSON, nullable=True)
    error: Mapped[str] = mapped_column(Text, nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_now)
    started_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=True)

    def to_dict(self) -> dict:
        return {
            "job_id": self.id,
            "status": self.status,
            "filename": self.filename,
            "image_id": self.image_id,
            "executor": self.executor,
            "external_run_id": self.external_run_id,
            "summary": self.summary,
            "error": self.error,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }


class RequestLog(Base):
    __tablename__ = "request_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    request_id: Mapped[str] = mapped_column(String(36), index=True)
    method: Mapped[str] = mapped_column(String(8))
    path: Mapped[str] = mapped_column(String(512))
    status_code: Mapped[int] = mapped_column(Integer)
    latency_ms: Mapped[float] = mapped_column(Float)
    client: Mapped[str] = mapped_column(String(128), nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)


class Feedback(Base):
    __tablename__ = "feedback"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(String(36), ForeignKey("jobs.id"), index=True)
    rating: Mapped[int] = mapped_column(Integer, nullable=True)
    comment: Mapped[str] = mapped_column(Text, nullable=True)
    # Corrected annotations in the spec JSON format; harvested into training data.
    corrections: Mapped[dict] = mapped_column(JSON, nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_now)


@lru_cache
def _engine(dsn: str):
    return create_engine(dsn, pool_pre_ping=True, future=True)


def engine():
    return _engine(get_settings().postgres_dsn)


@lru_cache
def _sessionmaker(dsn: str):
    return sessionmaker(bind=_engine(dsn), expire_on_commit=False, future=True)


@contextmanager
def session():
    sess = _sessionmaker(get_settings().postgres_dsn)()
    try:
        yield sess
        sess.commit()
    except Exception:
        sess.rollback()
        raise
    finally:
        sess.close()


def init_db() -> None:
    Base.metadata.create_all(engine())
