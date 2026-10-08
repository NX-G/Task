"""Job orchestrator task. Runs the pipeline inline or hands it to Airflow and
waits, and owns the job's lifecycle in Postgres either way."""

import datetime as dt
import logging
import time

import httpx
from celery.signals import worker_process_init

from drawing_ai import pipeline
from drawing_ai.config import get_settings
from drawing_ai.models import get_model_suite

from .. import db
from .celery_app import celery_app

log = logging.getLogger(__name__)


@worker_process_init.connect
def _warm_up(**_):
    db.init_db()
    s = get_settings()
    if s.pipeline_executor == "inline":
        get_model_suite(s)  # load models once per worker process


def _set(job_id: str, **fields):
    with db.session() as sess:
        job = sess.get(db.Job, job_id)
        for k, v in fields.items():
            setattr(job, k, v)


@celery_app.task(name="service.worker.tasks.process_job", bind=True, max_retries=0)
def process_job(self, job_id: str):
    s = get_settings()
    with db.session() as sess:
        job = sess.get(db.Job, job_id)
        if job is None:
            raise ValueError(f"Unknown job {job_id}")
        file_id, filename, image_id = job.input_file_id, job.filename, job.image_id
    _set(job_id, status=db.JobStatus.RUNNING, started_at=dt.datetime.now(dt.timezone.utc), executor=s.pipeline_executor)
    try:
        if s.pipeline_executor == "airflow":
            summary = _run_airflow(job_id, file_id, filename, image_id, s)
        else:
            summary = pipeline.run_inline(job_id, file_id, filename, image_id, s)
        _set(job_id, status=db.JobStatus.SUCCEEDED, summary=summary, finished_at=dt.datetime.now(dt.timezone.utc))
        return summary
    except Exception as exc:
        log.exception("Job %s failed", job_id)
        _set(job_id, status=db.JobStatus.FAILED, error=f"{type(exc).__name__}: {exc}", finished_at=dt.datetime.now(dt.timezone.utc))
        raise


def _run_airflow(job_id, file_id, filename, image_id, s) -> dict:
    auth = (s.airflow_user, s.airflow_password)
    run_id = f"job_{job_id}"
    conf = {"job_id": job_id, "input_file_id": file_id, "filename": filename, "image_id": image_id}
    with httpx.Client(base_url=s.airflow_api_url, auth=auth, timeout=30) as http:
        r = http.post(f"/dags/{s.airflow_dag_id}/dagRuns", json={"dag_run_id": run_id, "conf": conf})
        if r.status_code != 409:  # 409 = already triggered (task redelivery)
            r.raise_for_status()
        _set(job_id, external_run_id=run_id)
        deadline = time.monotonic() + s.job_timeout_s
        while time.monotonic() < deadline:
            state = http.get(f"/dags/{s.airflow_dag_id}/dagRuns/{run_id}").json()["state"]
            if state == "success":
                return {"airflow_run_id": run_id, "state": state}
            if state == "failed":
                raise RuntimeError(f"Airflow run {run_id} failed")
            time.sleep(s.airflow_poll_s)
    raise TimeoutError(f"Airflow run {run_id} did not finish in {s.job_timeout_s}s")
