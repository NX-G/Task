"""Central configuration. Every value can be overridden by an env var of the same
name (upper-case), e.g. ``TILE_SIZE=768``."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- Geometry: every drawing is normalised to one fixed canvas, then tiled.
    canvas_width: int = 4096
    canvas_height: int = 2896  # ISO A-series aspect ratio (1:1.414)
    tile_size: int = 1024
    tile_overlap: int = 128
    pdf_dpi: int = 300
    max_pages: int = 10

    # --- Stage 1 model backends: "auto" uses the trained model when it can be
    # resolved from MLflow and its libraries are installed, otherwise the
    # lightweight OpenCV / Tesseract fallback.
    detector_backend: str = "auto"  # auto | yolo | heuristic
    line_backend: str = "auto"  # auto | rfdetr | hough
    ocr_backend: str = "auto"  # auto | got_ocr2 | tesseract | none
    device: str = "cpu"

    # --- Post-processing thresholds
    detector_conf: float = 0.25
    line_conf: float = 0.30
    ocr_conf: float = 0.30

    # --- Model registry (MLflow)
    mlflow_tracking_uri: str = "http://mlflow:5000"
    yolo_model_name: str = "component-detector"
    rfdetr_model_name: str = "annotation-line-detector"
    model_alias: str = "champion"
    got_ocr_model_id: str = "stepfun-ai/GOT-OCR-2.0-hf"
    model_cache_dir: str = "/models"

    # --- Stage 2 patch network (LLM merge). Any OpenAI-compatible chat
    # completions endpoint: Azure AI Foundry, HF TGI / vLLM, OpenAI.
    merge_backend: str = "auto"  # auto | llm | heuristic
    llm_base_url: str = ""
    llm_api_key: str = ""
    llm_api_version: str = ""  # set for Azure (adds ?api-version=)
    llm_model: str = "gpt-4o-mini"
    llm_timeout_s: float = 60.0
    prompt_version: str = "patch_merge_v1"

    # --- Stage 2 heuristic tolerances (pixels in canvas space)
    merge_seam_margin: int = 48
    merge_endpoint_tol: float = 24.0
    merge_line_dist_tol: float = 6.0
    merge_angle_tol_deg: float = 5.0
    merge_overlap_iou: float = 0.5

    # --- Storage / infra
    mongo_uri: str = "mongodb://mongo:27017"
    mongo_db: str = "drawing_ai"
    postgres_dsn: str = "postgresql+psycopg2://drawing:drawing@postgres:5432/drawing_ai"
    redis_url: str = "redis://redis:6379/0"

    # --- Orchestration
    pipeline_executor: str = "inline"  # inline | airflow
    airflow_api_url: str = "http://airflow-webserver:8080/api/v1"
    airflow_user: str = "airflow"
    airflow_password: str = "airflow"
    airflow_dag_id: str = "drawing_inference"
    airflow_poll_s: float = 3.0
    job_timeout_s: int = 1800

    # --- API gateway
    api_key: str = ""  # empty = auth disabled
    max_upload_mb: int = 50

    # --- Drift detection
    drift_psi_threshold: float = 0.25
    drift_rolling_window: int = 50


@lru_cache
def get_settings() -> Settings:
    return Settings()
