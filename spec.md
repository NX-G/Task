# Drawing AI: Engineering-Drawing Understanding System (spec.md)

The source of truth for future work on this codebase. It is derived from
`SPEC/Algorithm.pdf` (the algorithm) and `SPEC/System Design - Google Slides.pdf`
(the architecture). It records every decision the code depends on. **If you change
behaviour, update this file in the same change.**

---

## 1. Goal

Take a 2D/3D engineering drawing (PDF or raster) and return structured JSON that lists:

- **Components / objects**: a COCO bounding box and a component class (`Ctype_*`).
- **OCR text**: a text box and the extracted text (dimensions, notes, symbols such as `Ø25.4`).
- **Annotation arrows / lines**: segment endpoints and a line type (`leader_line`, `section_arrow`, `dimension_line`, …).

The system has two parts: an **offline training** stack and an **online inference** stack. Both run in Docker.

---

## 2. Algorithm (from `SPEC/Algorithm.pdf`)

### 2.1 Fixed dimensions + tiling
1. Render the input. A PDF is rendered at `PDF_DPI` (300) with PyMuPDF. A raster image is decoded with OpenCV. Each page is processed on its own.
2. **Orientation**: a portrait page is rotated 90° clockwise. Sheets are treated as landscape.
3. **Fixed dimension**: the page is letterboxed onto one fixed canvas,
   `CANVAS_WIDTH × CANVAS_HEIGHT` = **4096 × 2896** (ISO A aspect). It is scaled to fit and padded with white, centred.
   `CanvasTransform` (scale, pad, rotation) maps canvas coordinates to original coordinates in both directions.
4. **Enhancement**: grayscale plus a min-max contrast stretch, returned as 3-channel BGR. There is deliberately no blur, so 1-px lines survive.
5. **Tiles**: `TILE_SIZE` = 1024 with `TILE_OVERLAP` = 128 (stride 896). The last row and column are flush with the canvas edge.
   The grid is deterministic and row-major. `Tile_001` is top-left, and the default grid is 5 cols × 4 rows = 20 tiles.
   Training and inference **must** use identical geometry settings, or Tile IDs and coordinates will not line up.

### 2.2 Training-data format (per tile, tile-local pixel coordinates)
```json
{
  "Image ID": "IMG_001",
  "Tile ID": "Tile_001",
  "OBJECT/COMPONENT": [{"object_Bounding_Box": [120, 340, 85, 42], "classification_type": "Ctype_1"}],
  "OCR":              [{"content_bbox": [122, 342, 80, 38], "text_extracted": "Ø25.4"}],
  "Annotation Arrow": [{"Points": [205, 361, 280, 361], "class_id": "leader_line"}]
}
```
- Boxes use COCO `[x, y, width, height]`. Points are `[x1, y1, x2, y2]`.
- Pydantic models live in `src/drawing_ai/schemas.py`. The aliases are the spec keys, and the Python names are `bbox` / `text` / `points`.
- Optional extensions on output items: `confidence` and `source_tiles`. Optional extension on a tile: `tile_box` (`[x, y, w, h]` on the canvas).
- `ImageAnnotation` has the same keys without `Tile ID`, in original-image pixels. Feature engineering slices it into tiles.

### 2.3 Stage 1: meta extraction (every tile → 3 models)
| Model | Purpose | Trained backend | Fallback (no trained model) |
|---|---|---|---|
| Model 1 | Component detection | **YOLO** (Ultralytics), MLflow `component-detector@champion` | `HeuristicDetector`: large closed contours, labelled `component` |
| Model 2 | Annotation-line detection | **RF-DETR**, MLflow `annotation-line-detector@champion` | `HoughLineDetector`: thin-stroke HoughLinesP, labelled `line_candidate` |
| Model 3 | OCR | **GOT-OCR2.0** (`stepfun-ai/GOT-OCR-2.0-hf`) on proposed text regions | `TesseractOCR` (psm 11, words grouped into lines) |

- The backend is chosen by `DETECTOR_BACKEND` / `LINE_BACKEND` / `OCR_BACKEND`.
  With `auto`, the system uses the trained model when it resolves from MLflow and its libraries are installed. Otherwise it logs a warning and uses the fallback.
  Naming the model explicitly (`yolo` / `rfdetr` / `got_ocr2`) makes a missing model a hard error.
- **Lines as boxes (RF-DETR)**: RF-DETR detects boxes. For training, a segment becomes a padded box (`geometry.segment_to_box`, pad 4 px).
  At inference the box is decoded back (`geometry.box_to_segment`):
  - If `w ≥ 3h`, the line is horizontal through the box centre.
  - If `h ≥ 3w`, the line is vertical.
  - Otherwise it is a diagonal. The direction ("down" or "up") is whichever diagonal has more ink in the tile.
- **GOT-OCR2.0 does not localise text**. Regions are proposed by Tesseract word boxes, or by a morphological text detector (`detect_text_regions`) if Tesseract is absent.
  Crops are padded by 3 px and transcribed in batches. GOT has no calibrated score, so items get a confidence prior of 0.9.
- The model suite is built once per process (`models.factory.get_model_suite`). `ModelSuite.versions` is recorded in every result.

### 2.4 Stage 2: patch network (LLM merge)
Implementation: `src/drawing_ai/merge/`.
1. All tile detections are converted to canvas coordinates as `Candidate`s, with uid `"<Tile ID>:<kind>:<index>"`.
2. **Pairs**: left/right neighbours (vertical seam) and top/bottom neighbours (horizontal seam).
   Diagonal neighbours are not paired; they are joined transitively.
3. For each pair, the candidates inside the seam region (tile overlap expanded by `MERGE_SEAM_MARGIN` = 48 px) are collected.
   The pair is skipped unless both sides have candidates.
4. **Heuristic decision** (`merge/heuristic.py`). It is always computed:
   - **object**: IoU of the boxes clipped to the overlap region ≥ `MERGE_OVERLAP_IOU` (0.5), or full-box IoU ≥ 0.5.
   - **arrow**: angle difference ≤ 5°, perpendicular distance ≤ 6 px, and endpoint gap ≤ 24 px.
   - **ocr**: clipped IoU ≥ 0.5, or (vertical seam only) same baseline, similar height and horizontal gap ≤ 24 px.
5. **LLM decision** (`merge/llm.py`). This runs when `LLM_BASE_URL` is set and `MERGE_BACKEND` is `auto` or `llm`:
   - It calls an OpenAI-compatible `/chat/completions` endpoint (Azure AI Foundry, HF TGI/vLLM or OpenAI) with `temperature=0` and `response_format=json_object`.
   - The system prompt is the versioned template `src/drawing_ai/prompts/<PROMPT_VERSION>.txt`.
   - The user message is JSON with the seam info, both tiles' items (id, kind, geom, label, confidence) and the heuristic suggestions as hints.
   - The expected reply is `{"merges":[{"ids":[...],"label":"<optional>"}],"relabels":[{"id":...,"label":...}]}`.
   - The reply is **validated** (`parse_decision`): unknown ids and mixed-kind groups are dropped.
   - On any LLM error, that pair falls back to the heuristic decision. `merge_report.llm_failures` counts these fallbacks.
6. **Union-find** over every pair's merges collapses an entity that spans 3 or more tiles into one group.
7. **Fusion** per group:
   - **object**: union box. The class is a confidence-weighted vote, unless the LLM gave a label.
   - **arrow**: a collinear merge, the longest span along the reference direction. The class is a vote, unless the LLM gave a label.
   - **ocr**: union box. The text is joined in reading order: one fragment containing the other, or a suffix/prefix overlap (`"Ø25." + "25.4" → "Ø25.4"`). A duplicate read of the same region keeps the longer reading. An LLM label overrides all of this.
   - Every fused item carries `source_tiles` and the max confidence.
8. **Final JSON** = the per-tile JSON plus the combined merged JSON (§5).

### 2.5 Post-processing
- **Thresholds**: `DETECTOR_CONF`, `LINE_CONF` and `OCR_CONF` are applied per item before merging. Items without a confidence (ground truth) are kept.
- **Merge tiles**: Stage 2, above.
- **Visualization**: on the canvas, components are green boxes, OCR is blue boxes with text and arrows are red lines with a start dot. The image is downscaled to at most 4096 px and stored as PNG in GridFS.
- **Drift detection**: see §7.

---

## 3. Architecture (from `SPEC/System Design`)

```
                                OFFLINE TRAINING (Docker: trainer, Azure VM)
 Storage (Mongo + artifacts) ─▶ Feature Engineering ─▶ Model Training ─▶ Post Processing
                                         │                  │                │ model + prompt versions
                                         ▼                  ▼                ▼
                               Container Registry     On-prem/cloud VLM   Model Registry
                                    (Azure ACR)       (AI Foundry / HF)     (MLflow)
                                         │                  │                │
 INFERENCE                               ▼                  ▼                ▼
 API (FastAPI) ◀─▶ Job orchestrator ─▶ Preprocessing ─▶ Inference ─▶ Post Processing
   │               (Redis + Celery)      pipeline        pipeline    (threshold, merge, viz, drift)
   ▼                     ▲              (Celery inline or Airflow DAG)        │
 Postgres            Storage (Mongo: inference + eval data) ◀──────────────────┘
 (jobs, logs, feedback)
```

| Box in diagram | Implementation |
|---|---|
| API (AI Gateway + FastAPI) | `src/Deployment/service/api/` with X-API-Key auth, request ids and request logging |
| DB Postgres (log + feedback) | `service/db.py`: `jobs`, `request_logs`, `feedback` |
| Job orchestrator (Redis + Celery) | `service/worker/`: queue `drawings`, task `process_job` |
| Preprocessing / Inference / Post-processing pipelines (Airflow + Python) | `drawing_ai/pipeline.py` stage functions, plus the DAG `src/Deployment/airflow/dags/drawing_inference_dag.py` |
| Storage (artifacts + MongoDB) | `drawing_ai/storage/mongo.py`: GridFS plus collections |
| Model Registry (MLflow) | `mlflow` service. Registered models are `component-detector` and `annotation-line-detector`, with alias `champion` |
| On-prem / cloud VLM | GOT-OCR2.0 from Hugging Face. The LLM is any OpenAI-compatible endpoint (Azure AI Foundry, TGI) |
| Container Registry (Azure) | `scripts/push_acr.sh` |
| Offline training (Docker on Azure VM) | `src/Training/` (image `drawing-ai/trainer`) |

### 3.1 Request lifecycle
1. `POST /v1/jobs` (multipart `file`, optional `image_id`):
   - Validates the extension and size (`MAX_UPLOAD_MB`).
   - Stores the file in GridFS and inserts a `jobs` row (`queued`).
   - Sends a Celery task. Returns `202 {job_id}`.
2. The Celery `process_job` task marks the job `running`. Then:
   - With `PIPELINE_EXECUTOR=inline`, it runs `pipeline.run_inline`, which calls preprocess, then infer, then postprocess in the worker.
   - With `PIPELINE_EXECUTOR=airflow`, it triggers DAG `drawing_inference` through the Airflow REST API. The `dag_run_id` is `job_<job_id>`, and a 409 on redelivery is treated as already-triggered. It then polls until the run succeeds or fails, or `JOB_TIMEOUT_S` expires.
   - Finally it marks the job `succeeded` (with a summary) or `failed` (with an error).
3. Stage state is handed between stages through Mongo `pipeline_state`, so every stage is idempotent and can run on a different machine.
4. Clients poll `GET /v1/jobs/{id}`, then fetch `/result` and `/visualization`.

---

## 4. Repository layout

```
spec.md                     this file
docker-compose.yml          whole stack; profiles: (default) inference, airflow, training
.env.example                every tunable (maps 1:1 to drawing_ai.config.Settings)
Makefile                    up / up-airflow / test / smoke / train-smoke / train / build-ml / push
scripts/smoke_test.sh       upload → poll → save result.json + visualization.png
scripts/push_acr.sh         tag + push images to Azure Container Registry
tests/                      pytest unit tests (run inside the service image: `make test`)
SPEC/                       original spec PDFs
src/
  sample_data/              sample drawing PDF
  drawing_ai/               SHARED CORE (used by training, worker, Airflow)
    config.py               Settings (pydantic-settings, env-driven)
    schemas.py              spec JSON contracts
    geometry.py             box/segment math, line<->box encoding, clipping
    io.py                   PDF/image loading
    preprocessing.py        orientation, fixed canvas, CanvasTransform, tile grid
    models/                 detector.py (YOLO), lines.py (RF-DETR), ocr.py (GOT/Tesseract),
                            registry.py (MLflow resolution), factory.py (backend selection)
    inference.py            Stage 1
    merge/                  Stage 2 patch network (candidates, heuristic, llm, patch_network)
    prompts/                versioned LLM prompt templates
    postprocess.py          thresholds, merge, visualization, final JSON
    drift.py                profiles, PSI comparison, rolling baseline
    pipeline.py             stage_preprocess / stage_infer / stage_postprocess / run_inline
    storage/mongo.py        GridFS + collections
  Deployment/
    Dockerfile              API + worker image (INSTALL_ML build arg)
    requirements*.txt       core / service / ML extras
    service/api/            FastAPI app + gateway middleware
    service/worker/         Celery app + tasks
    service/db.py           SQLAlchemy models (Postgres)
    airflow/                Airflow image + DAG
    mlflow/                 MLflow server image
    postgres/               init script (creates mlflow + airflow DBs)
  Training/
    Dockerfile              training image (torch CPU by default; TORCH_INDEX for CUDA)
    trainer/                synthetic, feature_engineering, train_yolo, train_rfdetr,
                            post_processing, mongo_source, tracking, pipeline (CLI)
```

Import paths: `PYTHONPATH` contains `src` (for `drawing_ai`), `src/Deployment` (for `service`) and `src/Training` (for `trainer`).

---

## 5. API contract

Every `/v1/*` route requires the `X-API-Key` header when `API_KEY` is set. Each response carries `X-Request-ID`.

| Method | Path | Description |
|---|---|---|
| GET | `/healthz` | liveness |
| GET | `/readyz` | postgres / mongo / redis checks (503 if any fails) |
| POST | `/v1/jobs` | multipart `file` (pdf, png, jpg, jpeg, tif, tiff, bmp, webp), optional `image_id` → `202 {job_id, status}` |
| GET | `/v1/jobs?limit=&status=` | recent jobs |
| GET | `/v1/jobs/{id}` | job status (`queued` / `running` / `succeeded` / `failed`), summary, error, timestamps |
| GET | `/v1/jobs/{id}/result?view=full\|merged` | final JSON (409 until succeeded) |
| GET | `/v1/jobs/{id}/visualization?page=0` | PNG with boxes drawn |
| POST | `/v1/jobs/{id}/feedback` | `{rating 1-5, comment, corrections:[spec JSON]}` → stored in Postgres |

**Result document** (Mongo `results`, returned by `/result`):
```json
{
  "job_id": "...", "filename": "...",
  "model_versions": {"detector": "yolo:3", "line_detector": "rfdetr:2", "ocr": "got-ocr2:..."},
  "settings": {"canvas": [4096, 2896], "tile_size": 1024, "tile_overlap": 128},
  "timings": {"preprocess_s": 0.4, "infer_s": 4.5, "postprocess_s": 0.4},
  "pages": [{
    "page": 0, "Image ID": "IMG_001",
    "canvas": {"orig_w": 0, "orig_h": 0, "rotated": false, "scale": 0, "pad_x": 0, "pad_y": 0, "canvas_w": 4096, "canvas_h": 2896},
    "tiles": ["<per-tile spec JSON, tile-local coords, + tile_box>"],
    "merged": {"OBJECT/COMPONENT": [], "OCR": [], "Annotation Arrow": []},
    "merged_original_coords": {"…same, in uploaded-image pixels…": []},
    "merge_report": {"backend": "llm|heuristic", "prompt_version": "...", "pairs": 0, "llm_calls": 0, "llm_failures": 0, "merged_groups": 0},
    "counts": {"OBJECT/COMPONENT": 0, "OCR": 0, "Annotation Arrow": 0},
    "drift": {"status": "ok|no_baseline", "drift": false, "metrics": {}, "flags": []},
    "visualization_file_id": "..."
  }]
}
```
A multi-page PDF yields one entry per page (up to `MAX_PAGES`). For a multi-page input the Image ID is `<image_id>_P<n>`.

---

## 6. Offline training

CLI: `python -m trainer.pipeline <command>`. The container entrypoint already includes `python -m trainer.pipeline`.

| Command | Does |
|---|---|
| `synth --raw /data/raw --n 30` | Generates synthetic drawings with exact ground truth (classes Ctype_1-3, dimension_line, leader_line, section_arrow, dimension text) |
| `export --raw /data/raw` | Pulls `training_images` (GridFS image + annotation) from MongoDB |
| `features --raw … --dataset …` | Feature engineering (below) |
| `train-yolo --dataset …` | Model 1 |
| `train-rfdetr --dataset …` | Model 2 |
| `post --dataset … --yolo-run <id> --rfdetr-run <id>` | Post-processing (below) |
| `all [--synthetic N] [--epochs E] [--skip-rfdetr]` | Everything above, in order |

**Raw layout**: `images/<IMAGE_ID>.<ext>` and `annotations/<IMAGE_ID>.json`. The annotation is either image-level (original pixels) or a list of tile annotations (tile-local).

**Feature engineering** (`feature_engineering.py`) runs the same `preprocess()` as inference, then:
- **Slicing image-level ground truth into tiles**:
  - Objects are kept if at least 30% of their area is visible, clipped to the tile.
  - OCR items are kept only if at least 90% is visible, since a partial text box would carry a wrong label.
  - Arrows are clipped with Liang-Barsky and kept if at least 10 px long.
- **Empty tiles** are kept with probability `empty_ratio` (0.2) as negatives.
- **Splits** are made per *image* (md5 hash → 80/10/10), never per tile. Valid and test are forced non-empty when there are at least 3 images.
- **Outputs**:
  - `tiles/`, plus `annotations/` (spec JSON).
  - `yolo/` (data.yaml, normalised labels, classes.json).
  - `coco_lines/{train,valid,test}/_annotations.coco.json`. This is Roboflow-style: category 0 is a placeholder and the classes are 1..N. `classes.json` is indexed by category id.
  - `ocr/<split>.jsonl` plus crops.
  - `manifest.json`.

**Training**:
- **YOLO**: `yolo11n.pt` base, `imgsz = TILE_SIZE`. Ultralytics' own MLflow callback is disabled.
- **RF-DETR**: `RFDETRBase`, metrics parsed from `log.txt`.
- **Logging**: each training run logs params, metrics and `model/` artifacts (weights plus `classes.json`) to MLflow experiment `drawing-ai`.

**Post-processing** (`post_processing.py`):
1. **OCR evaluation**: CER and exact match of the configured OCR backend on the validation crops (run `ocr-eval`).
2. **Register and promote**: `register_model(runs:/<id>/model)`. The alias `champion` moves only if the selection metric beats the current champion:
   - YOLO: `metrics/mAP50-95_B`.
   - RF-DETR: `mAP50_95`.
3. **Prompt versions**: the `prompts/*.txt` templates are logged as artifacts of a `prompt-versions` run, tagged `prompt_version`.
4. **Drift baseline**: the ground-truth profile of the validation split goes to Mongo `baselines` and the MLflow artifact `baseline.json`.

Inference picks up new champions on worker restart. Weights are cached in `/models/<name>/<version>`.

---

## 7. Drift detection

- A **profile** (`drift.profile`) is computed over the tiles of one page after thresholding. It contains:
  - The class distributions of objects and arrows.
  - Objects, arrows and OCR items per tile.
  - The mean confidence.
- **Baseline**: the latest Mongo `baselines` entry named `default`, written by training. If there is none, the rolling average of the last `DRIFT_ROLLING_WINDOW` (50) inference profiles is used. If that is empty too, the status is `no_baseline`.
- **Flags**:
  - Class-distribution PSI > `DRIFT_PSI_THRESHOLD` (0.25).
  - A per-tile rate shift > 1.0. This is symmetric relative change, so 1.0 ≈ a 3× change.
  - A mean-confidence drop > 0.15.
- Each comparison is stored in Mongo `drift` and returned in the result.

---

## 8. Storage

**MongoDB** (`MONGO_DB`, default `drawing_ai`):

| Collection | Content |
|---|---|
| GridFS `files` | inputs (`kind=input`), canvases (`kind=canvas`), visualizations (`kind=visualization`) |
| `pipeline_state` | per-job stage hand-off: pages, tile layout, transform, raw tile annotations, model versions, timings |
| `results` | final JSON (inference + eval data) |
| `drift` | per-page profile + comparison |
| `baselines` | drift baselines from training |
| `training_images` | optional training source: `{image_id, file_id, filename, annotation}` |

**Postgres** has three databases:
- `drawing_ai`: tables `jobs`, `request_logs` and `feedback`.
- `mlflow`: the MLflow backend store.
- `airflow`: the Airflow metadata.

The tables are created with `create_all` on startup.

**MLflow artifacts** are proxied by the server (`--serve-artifacts`) into the `mlartifacts` volume.

---

## 9. Configuration
All settings are in `src/drawing_ai/config.py` and documented in `.env.example`. Each one is overridable by an upper-case env var. Key groups:
- **Geometry**: `CANVAS_*`, `TILE_*`, `PDF_DPI`.
- **Backends**: `*_BACKEND`, `DEVICE`.
- **Thresholds**.
- **Registry**: `MLFLOW_TRACKING_URI`, model names, alias.
- **LLM**: `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_API_VERSION`, `LLM_MODEL`, `PROMPT_VERSION`.
- **Merge tolerances**.
- **Infra URIs**.
- **Orchestration**: `PIPELINE_EXECUTOR`, the Airflow settings, `JOB_TIMEOUT_S`.
- **Gateway**: `API_KEY`, `MAX_UPLOAD_MB`.
- **Drift**.

---

## 10. Running (Docker Desktop)

```bash
cp .env.example .env                 # optional
make up                              # postgres, mongo, redis, mlflow, api, worker
make smoke                           # sample PDF end-to-end → data/smoke/
make test                            # unit tests in the service image
make up-airflow                      # + Airflow (UI http://localhost:8080, airflow/airflow), PIPELINE_EXECUTOR=airflow
make train-smoke                     # tiny synthetic training run → registers champions in MLflow
INSTALL_ML=true make build-ml && docker compose up -d worker   # use the trained YOLO/RF-DETR/GOT models
```

| URL | Service |
|---|---|
| http://localhost:8000/docs | API (OpenAPI UI) |
| http://localhost:5001 | MLflow. Host port 5001, because macOS AirPlay uses 5000 |
| http://localhost:8080 | Airflow (with the airflow profile) |

The default service image has **no torch**: it is about 600 MB and runs the fallback models.
- Build with `INSTALL_ML=true` to get the YOLO, RF-DETR and GOT-OCR2.0 backends.
- On a GPU Azure VM, set `DEVICE=cuda` and build the trainer with `TORCH_INDEX=https://download.pytorch.org/whl/cu124`. The compose services also need GPU reservations.

**Deploy to Azure**: `ACR_NAME=<registry> TAG=v1 make push`. Then on the Azure VM, pull the images and run the same compose file with the registry image names, pointing `.env` at managed Postgres, Mongo (Cosmos DB for MongoDB vCore) and Redis if desired.

---

## 11. Known constraints and decisions
- **The fallback models are not accurate**. They exist so the whole system runs and is testable before training. Production needs trained YOLO and RF-DETR champions, plus GOT-OCR2.0 (`INSTALL_ML=true`).
- **Pinned versions**:
  - Airflow 2.10.5 needs numpy < 2, so the Airflow image pins `numpy<2` and `opencv-python-headless<4.11`.
  - MLflow 2.x needs SQLAlchemy < 2.1.
- **Airflow runs with LocalExecutor**, so tasks run in the scheduler container, which shares the `modelcache` volume.
- **Celery uses `task_acks_late` and `prefetch=1`**: a crashed worker's job is redelivered. Pipeline stages are idempotent (upserts keyed by job_id).
- **There are no DB migrations yet** (`create_all`). Add Alembic before changing table schemas in production.

## 12. Roadmap / next steps
1. A feedback → training loop: export `feedback.corrections` into `training_images`.
2. Fine-tune GOT-OCR2.0 on the drawing-symbol vocabulary (Ø, ±, GD&T). Today it is evaluated only (CER).
3. A tile-classification head (the "Tile Classification" box in the inference pipeline) to skip empty or title-block tiles before Stage 1.
4. Batched GPU inference across tiles, and a separate GPU worker queue.
5. Rate limiting and per-client quotas in the gateway. Prometheus metrics.
6. An Alembic migrations / Airflow 3 upgrade path.
