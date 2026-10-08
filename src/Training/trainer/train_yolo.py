"""Model 1 training: YOLO component detector on tile crops."""

import json
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Tuple

import mlflow

from .tracking import clean_metrics

log = logging.getLogger(__name__)


def train_yolo(dataset_dir: Path, work_dir: Path, epochs: int, imgsz: int, batch: int, base_weights: str, device: str) -> Tuple[str, dict]:
    from ultralytics import YOLO
    from ultralytics import settings as ysettings

    ysettings.update({"mlflow": False})  # we log to MLflow ourselves
    data_yaml = Path(dataset_dir) / "yolo" / "data.yaml"
    classes = json.loads((Path(dataset_dir) / "yolo" / "classes.json").read_text())
    if not classes:
        raise ValueError("No object classes in dataset; nothing to train")

    with mlflow.start_run(run_name="yolo-component-detector") as run:
        mlflow.set_tag("model_kind", "component_detector")
        mlflow.log_params({"base_weights": base_weights, "epochs": epochs, "imgsz": imgsz, "batch": batch, "classes": ",".join(classes)})
        mlflow.log_artifact(str(Path(dataset_dir) / "manifest.json"), "dataset")
        model = YOLO(base_weights)
        results = model.train(data=str(data_yaml), epochs=epochs, imgsz=imgsz, batch=batch, device=device,
                              project=str(work_dir), name="yolo", exist_ok=True, plots=False, verbose=False)
        metrics = clean_metrics(getattr(results, "results_dict", {}) or {})
        mlflow.log_metrics(metrics)
        with tempfile.TemporaryDirectory() as tmp:
            shutil.copy(model.trainer.best, Path(tmp) / "best.pt")
            (Path(tmp) / "classes.json").write_text(json.dumps(classes))
            mlflow.log_artifacts(tmp, "model")
        log.info("YOLO run %s metrics %s", run.info.run_id, metrics)
        return run.info.run_id, metrics
