"""Training post-processing:

1. OCR evaluation (character error rate) of the configured OCR backend.
2. Register trained models and promote to the ``champion`` alias when they beat
   the current champion on the selection metric.
3. Version the LLM prompt templates in MLflow (Model, Prompt Versions).
4. Compute and publish the drift-detection baseline (MongoDB + MLflow).
"""

import json
import logging
import tempfile
from importlib import resources
from pathlib import Path
from typing import Optional

import cv2
import mlflow
from mlflow import MlflowClient

from drawing_ai import drift
from drawing_ai.config import Settings

from .feature_engineering import load_tile_annotations

log = logging.getLogger(__name__)


def _levenshtein(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def evaluate_ocr(dataset_dir: Path, s: Settings, split: str = "valid", limit: int = 500) -> dict:
    from drawing_ai.models.factory import build_model_suite

    path = Path(dataset_dir) / "ocr" / f"{split}.jsonl"
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()][:limit] if path.exists() else []
    if not rows:
        return {}
    s_ocr = s.model_copy(update={"detector_backend": "heuristic", "line_backend": "hough"})
    ocr = build_model_suite(s_ocr).ocr
    errors = chars = exact = 0
    for r in rows:
        img = cv2.imread(str(Path(dataset_dir) / r["crop"]))
        pred = " ".join(o.text for o in ocr.predict(img, 0.0)).strip()
        errors += _levenshtein(pred, r["text"])
        chars += max(1, len(r["text"]))
        exact += pred == r["text"]
    metrics = {"ocr_cer": errors / chars, "ocr_exact_match": exact / len(rows), "ocr_samples": len(rows)}
    with mlflow.start_run(run_name="ocr-eval"):
        mlflow.set_tag("model_kind", "ocr")
        mlflow.log_param("ocr_version", ocr.version)
        mlflow.log_metrics(metrics)
    return metrics


def promote(name: str, run_id: str, metric: str, alias: str = "champion", higher_is_better: bool = True) -> dict:
    client = MlflowClient()
    mv = mlflow.register_model(f"runs:/{run_id}/model", name)
    new = client.get_run(run_id).data.metrics.get(metric)
    try:
        champ = client.get_model_version_by_alias(name, alias)
        old = client.get_run(champ.run_id).data.metrics.get(metric)
    except Exception:
        champ, old = None, None
    better = champ is None or old is None or (new is not None and (new > old if higher_is_better else new < old))
    if better:
        client.set_registered_model_alias(name, alias, mv.version)
    client.set_model_version_tag(name, mv.version, "selection_metric", f"{metric}={new}")
    decision = {"model": name, "version": mv.version, "metric": metric, "new": new, "champion_before": old, "promoted": better}
    log.info("Promotion: %s", decision)
    return decision


def log_prompts(s: Settings) -> str:
    with mlflow.start_run(run_name="prompt-versions") as run:
        mlflow.set_tag("model_kind", "llm_prompts")
        mlflow.set_tag("prompt_version", s.prompt_version)
        mlflow.log_params({"prompt_version": s.prompt_version, "llm_model": s.llm_model})
        prompts = resources.files("drawing_ai.prompts")
        with tempfile.TemporaryDirectory() as tmp:
            for f in prompts.iterdir():
                if f.name.endswith(".txt"):
                    (Path(tmp) / f.name).write_text(f.read_text(encoding="utf-8"), encoding="utf-8")
            mlflow.log_artifacts(tmp, "prompts")
        return run.info.run_id


def publish_baseline(dataset_dir: Path, s: Settings, split: str = "valid") -> dict:
    tiles = load_tile_annotations(dataset_dir, split) or load_tile_annotations(dataset_dir)
    prof = drift.profile(tiles)
    with mlflow.start_run(run_name="drift-baseline"):
        mlflow.set_tag("model_kind", "drift_baseline")
        mlflow.log_dict(prof, "baseline.json")
    try:
        from drawing_ai.storage.mongo import get_store

        get_store(s.mongo_uri, s.mongo_db).save_baseline("default", prof, {"dataset": str(dataset_dir), "split": split})
    except Exception as exc:
        log.warning("Could not write baseline to MongoDB (%s); MLflow copy only", exc)
    return prof


def run_post_processing(dataset_dir: Path, s: Settings, yolo_run: Optional[str], rfdetr_run: Optional[str], eval_ocr: bool = True) -> dict:
    report = {}
    if eval_ocr:
        report["ocr"] = evaluate_ocr(dataset_dir, s)
    if yolo_run:
        report["yolo"] = promote(s.yolo_model_name, yolo_run, "metrics/mAP50-95_B", s.model_alias)
    if rfdetr_run:
        report["rfdetr"] = promote(s.rfdetr_model_name, rfdetr_run, "mAP50_95", s.model_alias)
    report["prompts_run"] = log_prompts(s)
    report["baseline"] = publish_baseline(dataset_dir, s)
    return report
