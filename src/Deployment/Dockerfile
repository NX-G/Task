# API + Celery worker image (same image, different command).
# Build context: repository root.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src:/app/src/Deployment

RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY src/Deployment/requirements.txt src/Deployment/requirements-service.txt src/Deployment/requirements-ml.txt /tmp/
RUN pip install -r /tmp/requirements.txt -r /tmp/requirements-service.txt

ARG INSTALL_ML=false
RUN if [ "$INSTALL_ML" = "true" ]; then \
        pip install --extra-index-url https://download.pytorch.org/whl/cpu -r /tmp/requirements-ml.txt; \
    fi

COPY src/drawing_ai /app/src/drawing_ai
COPY src/Deployment/service /app/src/Deployment/service

RUN useradd -m -u 1000 app && mkdir -p /models && chown app /models
USER app

EXPOSE 8000
CMD ["uvicorn", "service.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
