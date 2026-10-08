from celery import Celery

from drawing_ai.config import get_settings

_s = get_settings()

celery_app = Celery("drawing_ai", broker=_s.redis_url, backend=_s.redis_url, include=["service.worker.tasks"])
celery_app.conf.update(
    task_acks_late=True,  # a crashed worker's job is redelivered
    worker_prefetch_multiplier=1,  # jobs are long; don't hoard them
    task_track_started=True,
    task_time_limit=_s.job_timeout_s + 60,
    task_soft_time_limit=_s.job_timeout_s,
    result_expires=86400,
    task_default_queue="drawings",
)
