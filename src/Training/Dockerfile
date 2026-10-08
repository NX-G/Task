# syntax=docker/dockerfile:1
# Offline training container (Feature Engineering -> Model Training -> Post Processing).
# Build context: repository root. For GPU on the Azure VM build with
#   --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu124
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src:/app/src/Training

RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr libgl1 libglib2.0-0 git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
COPY src/Deployment/requirements.txt /app/src/Deployment/requirements.txt
COPY src/Training/requirements.txt /app/src/Training/requirements.txt
# pip cache mount: a flaky download resumes from cache on rebuild
RUN --mount=type=cache,target=/root/.cache/pip env -u PIP_NO_CACHE_DIR pip install --default-timeout=120 --retries 10 --extra-index-url ${TORCH_INDEX} -r /app/src/Training/requirements.txt

COPY src/drawing_ai /app/src/drawing_ai
COPY src/Training/trainer /app/src/Training/trainer

ENTRYPOINT ["python", "-m", "trainer.pipeline"]
CMD ["--help"]
