"""Model 2 training: RF-DETR annotation-line detector (lines encoded as boxes)."""

import json
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Tuple

import mlflow

from .tracking import clean_metrics

log = logging.getLogger(__name__)
COCO_AP_KEYS = ["mAP50_95", "mAP50", "mAP75", "AP_small", "AP_medium", "AP_large"]


def _read_metrics(out_dir: Path) -> dict:
    """RF-DETR appends one JSON line per epoch to log.txt."""
    log_file = out_dir / "log.txt"
    if not log_file.exists():
        return {}
    last = None
    for line in log_file.read_text().splitlines():
        try:
            last = json.loads(line)
        except json.JSONDecodeError:
            continue
    if not last:
        return {}
    ap = last.get("test_coco_eval_bbox") or []
    metrics = {k: v for k, v in zip(COCO_AP_KEYS, ap)}
    metrics.update({k: v for k, v in last.items() if isinstance(v, (int, float))})
    return clean_metrics(metrics)


def train_rfdetr(dataset_dir: Path, work_dir: Path, epochs: int, batch: int, grad_accum: int, lr: float, device: str) -> Tuple[str, dict]:
    from rfdetr import RFDETRBase

    coco_dir = Path(dataset_dir) / "coco_lines"
    classes = json.loads((coco_dir / "classes.json").read_text())
    if len(classes) <= 1:
        raise ValueError("No annotation-line classes in dataset; nothing to train")
    out_dir = Path(work_dir) / "rfdetr"
    out_dir.mkdir(parents=True, exist_ok=True)

    with mlflow.start_run(run_name="rfdetr-annotation-lines") as run:
        mlflow.set_tag("model_kind", "annotation_line_detector")
        mlflow.log_params({"epochs": epochs, "batch": batch, "grad_accum": grad_accum, "lr": lr, "classes": ",".join(classes)})
        model = RFDETRBase(device=device)
        model.train(dataset_dir=str(coco_dir), epochs=epochs, batch_size=batch, grad_accum_steps=grad_accum, lr=lr,
                    output_dir=str(out_dir), device=device, early_stopping=False)
        metrics = _read_metrics(out_dir)
        mlflow.log_metrics(metrics)
        ckpt = out_dir / "checkpoint_best_total.pth"
        if not ckpt.exists():
            ckpt = max(out_dir.glob("checkpoint*.pth"), key=lambda p: p.stat().st_mtime)
        with tempfile.TemporaryDirectory() as tmp:
            shutil.copy(ckpt, Path(tmp) / "checkpoint_best_total.pth")
            (Path(tmp) / "classes.json").write_text(json.dumps(classes))
            mlflow.log_artifacts(tmp, "model")
        log.info("RF-DETR run %s metrics %s", run.info.run_id, metrics)
        return run.info.run_id, metrics
