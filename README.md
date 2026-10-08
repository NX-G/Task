# Drawing AI

Turns engineering drawings into structured JSON: components, OCR text and annotation arrows. It works in two stages:
- **Stage 1** splits the drawing into tiles and runs three models on each tile: YOLO, RF-DETR and GOT-OCR2.0.
- **Stage 2** is a patch network that uses an LLM to stitch detections across the tile seams.

It runs as Docker services: FastAPI, Celery/Redis, Airflow, Postgres, MongoDB and MLflow.

**Read [spec.md](spec.md) first.** It is the design contract for all future work.

## Quick start (Docker Desktop)

```bash
make up          # postgres, mongo, redis, mlflow, api, worker
make smoke       # sends src/sample_data/3D Drawing Example.pdf through the pipeline -> data/smoke/
make test        # unit tests
```

- API docs: http://localhost:8000/docs
- MLflow: http://localhost:5001
- Airflow: http://localhost:8080 (`make up-airflow`)

## Train models

```bash
make train-smoke                 # synthetic data, 1 epoch: proves the training path end to end
make train                       # real data in ./data/raw/{images,annotations}
INSTALL_ML=true make build-ml && docker compose up -d worker   # serve the trained champions
```
