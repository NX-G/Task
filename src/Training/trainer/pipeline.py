"""Training CLI.

    python -m trainer.pipeline synth     --raw /data/raw --n 30
    python -m trainer.pipeline export    --raw /data/raw            # from MongoDB
    python -m trainer.pipeline features  --raw /data/raw --dataset /data/processed/v1
    python -m trainer.pipeline train-yolo   --dataset ... [--epochs 50]
    python -m trainer.pipeline train-rfdetr --dataset ... [--epochs 30]
    python -m trainer.pipeline post      --dataset ... --yolo-run <id> --rfdetr-run <id>
    python -m trainer.pipeline all       --raw /data/raw --dataset /data/processed/v1 [--synthetic 30]
"""

import argparse
import json
import logging
from pathlib import Path

from drawing_ai.config import get_settings

from . import tracking

log = logging.getLogger("trainer")


def main(argv=None):
    p = argparse.ArgumentParser(prog="trainer")
    p.add_argument("command", choices=["synth", "export", "features", "train-yolo", "train-rfdetr", "post", "all"])
    p.add_argument("--raw", type=Path, default=Path("/data/raw"))
    p.add_argument("--dataset", type=Path, default=Path("/data/processed/latest"))
    p.add_argument("--work", type=Path, default=Path("/data/runs"))
    p.add_argument("--n", "--synthetic", dest="n", type=int, default=0, help="generate N synthetic drawings first")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--yolo-epochs", type=int, default=50)
    p.add_argument("--rfdetr-epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--yolo-weights", default="yolo11n.pt")
    p.add_argument("--rfdetr-lr", type=float, default=1e-4)
    p.add_argument("--yolo-run")
    p.add_argument("--rfdetr-run")
    p.add_argument("--skip-rfdetr", action="store_true")
    p.add_argument("--skip-ocr-eval", action="store_true")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    s = get_settings()
    yolo_epochs = a.epochs or a.yolo_epochs
    rfdetr_epochs = a.epochs or a.rfdetr_epochs

    if a.command == "synth" or (a.command == "all" and a.n):
        from .synthetic import generate

        generate(a.raw, a.n or 30)
        log.info("Generated %d synthetic drawings in %s", a.n or 30, a.raw)
    if a.command == "export":
        from .mongo_source import export_from_mongo

        log.info("Exported %d images from MongoDB", export_from_mongo(a.raw, s))
    if a.command in ("features", "all"):
        from .feature_engineering import build_datasets

        print(json.dumps(build_datasets(a.raw, a.dataset, s), indent=2))
    if a.command in ("train-yolo", "train-rfdetr", "post", "all"):
        tracking.setup(s.mlflow_tracking_uri)
    yolo_run, rfdetr_run = a.yolo_run, a.rfdetr_run
    if a.command in ("train-yolo", "all"):
        from .train_yolo import train_yolo

        yolo_run, _ = train_yolo(a.dataset, a.work, yolo_epochs, s.tile_size, a.batch, a.yolo_weights, s.device)
    if a.command == "train-rfdetr" or (a.command == "all" and not a.skip_rfdetr):
        from .train_rfdetr import train_rfdetr

        rfdetr_run, _ = train_rfdetr(a.dataset, a.work, rfdetr_epochs, a.batch, 4, a.rfdetr_lr, s.device)
    if a.command in ("post", "all"):
        from .post_processing import run_post_processing

        report = run_post_processing(a.dataset, s, yolo_run, rfdetr_run, eval_ocr=not a.skip_ocr_eval)
        print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
