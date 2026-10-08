"""AI-gateway concerns: API-key auth, request ids, request logging to Postgres."""

import logging
import secrets
import time
import uuid

from fastapi import HTTPException, Request, Security
from fastapi.security import APIKeyHeader
from starlette.concurrency import run_in_threadpool
from starlette.middleware.base import BaseHTTPMiddleware

from drawing_ai.config import get_settings

from .. import db

log = logging.getLogger(__name__)
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
_UNLOGGED = {"/healthz", "/readyz"}


def require_api_key(key: str = Security(_api_key_header)) -> None:
    expected = get_settings().api_key
    if expected and not (key and secrets.compare_digest(key, expected)):
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")


def _write_log(entry: dict) -> None:
    try:
        with db.session() as sess:
            sess.add(db.RequestLog(**entry))
    except Exception as exc:  # logging must never break a request
        log.warning("request log write failed: %s", exc)


class RequestLogMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        rid = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        t0 = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            response.headers["X-Request-ID"] = rid
            return response
        finally:
            if request.url.path not in _UNLOGGED:
                await run_in_threadpool(_write_log, {
                    "request_id": rid,
                    "method": request.method,
                    "path": request.url.path[:512],
                    "status_code": status,
                    "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
                    "client": request.client.host if request.client else None,
                })
