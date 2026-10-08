"""FastAPI gateway: submit drawings, poll jobs, fetch results / visualisations,
send feedback."""

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import redis
from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, Field
from sqlalchemy import text

from drawing_ai.config import get_settings
from drawing_ai.io import SUPPORTED_EXTS
from drawing_ai.storage.mongo import get_store

from .. import db
from ..worker.celery_app import celery_app
from .gateway import RequestLogMiddleware, require_api_key

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(_: FastAPI):
    db.init_db()
    yield


app = FastAPI(title="Drawing AI", version="0.1.0", lifespan=lifespan)
app.add_middleware(RequestLogMiddleware)
v1 = [Depends(require_api_key)]


def _store():
    s = get_settings()
    return get_store(s.mongo_uri, s.mongo_db)


def _job_or_404(sess, job_id: str) -> db.Job:
    job = sess.get(db.Job, job_id)
    if job is None:
        raise HTTPException(404, f"Job {job_id} not found")
    return job


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.get("/readyz")
def readyz():
    s = get_settings()
    checks: Dict[str, Any] = {}
    for name, fn in {
        "postgres": lambda: db.engine().connect().execute(text("SELECT 1")).close(),
        "mongo": lambda: _store().ping(),
        "redis": lambda: redis.Redis.from_url(s.redis_url, socket_timeout=2).ping(),
    }.items():
        try:
            fn()
            checks[name] = "ok"
        except Exception as exc:
            checks[name] = f"error: {type(exc).__name__}"
    ok = all(v == "ok" for v in checks.values())
    if not ok:
        raise HTTPException(503, checks)
    return checks


@app.post("/v1/jobs", status_code=202, dependencies=v1)
async def submit_job(file: UploadFile = File(...), image_id: Optional[str] = Form(None)):
    s = get_settings()
    ext = Path(file.filename or "").suffix.lower()
    if ext not in SUPPORTED_EXTS:
        raise HTTPException(415, f"Unsupported file type {ext!r}; allowed: {sorted(SUPPORTED_EXTS)}")
    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty file")
    if len(data) > s.max_upload_mb * 1024 * 1024:
        raise HTTPException(413, f"File exceeds {s.max_upload_mb} MB")
    file_id = _store().put_file(data, file.filename, file.content_type or "application/octet-stream", kind="input")
    with db.session() as sess:
        job = db.Job(filename=file.filename, image_id=image_id, input_file_id=file_id, executor=s.pipeline_executor)
        sess.add(job)
        sess.flush()
        job_id = job.id
    celery_app.send_task("service.worker.tasks.process_job", args=[job_id])
    return {"job_id": job_id, "status": db.JobStatus.QUEUED}


@app.get("/v1/jobs", dependencies=v1)
def list_jobs(limit: int = Query(20, le=200), status: Optional[str] = None):
    with db.session() as sess:
        q = sess.query(db.Job).order_by(db.Job.created_at.desc())
        if status:
            q = q.filter(db.Job.status == status)
        return [j.to_dict() for j in q.limit(limit)]


@app.get("/v1/jobs/{job_id}", dependencies=v1)
def get_job(job_id: str):
    with db.session() as sess:
        return _job_or_404(sess, job_id).to_dict()


@app.get("/v1/jobs/{job_id}/result", dependencies=v1)
def get_result(job_id: str, view: str = Query("full", pattern="^(full|merged)$")):
    with db.session() as sess:
        job = _job_or_404(sess, job_id)
        if job.status != db.JobStatus.SUCCEEDED:
            raise HTTPException(409, f"Job is {job.status}")
    result = _store().get_result(job_id)
    if result is None:
        raise HTTPException(404, "Result not found")
    result.pop("created_at", None)
    if view == "merged":
        result["pages"] = [{k: p[k] for k in ("page", "Image ID", "merged", "counts", "drift")} for p in result["pages"]]
    return result


@app.get("/v1/jobs/{job_id}/visualization", dependencies=v1, response_class=Response)
def get_visualization(job_id: str, page: int = 0):
    result = _store().get_result(job_id)
    if result is None:
        raise HTTPException(404, "Result not found")
    pages = {p["page"]: p for p in result["pages"]}
    if page not in pages:
        raise HTTPException(404, f"Page {page} not found")
    return Response(_store().get_file(pages[page]["visualization_file_id"]), media_type="image/png")


class FeedbackIn(BaseModel):
    rating: Optional[int] = Field(None, ge=1, le=5)
    comment: Optional[str] = Field(None, max_length=5000)
    corrections: Optional[List[Dict[str, Any]]] = Field(
        None, description="Corrected annotations in the training JSON format (per tile or per image)."
    )


@app.post("/v1/jobs/{job_id}/feedback", status_code=201, dependencies=v1)
def post_feedback(job_id: str, body: FeedbackIn):
    with db.session() as sess:
        _job_or_404(sess, job_id)
        fb = db.Feedback(job_id=job_id, rating=body.rating, comment=body.comment,
                         corrections={"items": body.corrections} if body.corrections else None)
        sess.add(fb)
        sess.flush()
        return {"feedback_id": fb.id}
