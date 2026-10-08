"""Model 1 — component object detection (YOLO)."""

import json
from pathlib import Path
from typing import List

import cv2
import numpy as np

from ..schemas import ObjectComponent


class YoloDetector:
    """Ultralytics YOLO trained on tile crops (see trainer/train_yolo.py)."""

    def __init__(self, model_dir: Path, version: str, device: str = "cpu"):
        from ultralytics import YOLO

        self.model = YOLO(str(Path(model_dir) / "best.pt"))
        classes_file = Path(model_dir) / "classes.json"
        self.names = json.loads(classes_file.read_text()) if classes_file.exists() else dict(self.model.names)
        self.device = device
        self.version = f"yolo:{version}"

    def predict(self, img: np.ndarray, conf: float) -> List[ObjectComponent]:
        res = self.model.predict(img, imgsz=max(img.shape[:2]), conf=conf, device=self.device, verbose=False)[0]
        out = []
        for (x1, y1, x2, y2), c, p in zip(res.boxes.xyxy.tolist(), res.boxes.cls.tolist(), res.boxes.conf.tolist()):
            name = self.names[int(c)] if isinstance(self.names, list) else self.names.get(int(c), self.names.get(str(int(c))))
            out.append(ObjectComponent(bbox=[x1, y1, x2 - x1, y2 - y1], classification_type=str(name), confidence=float(p)))
        return out


class HeuristicDetector:
    """Untrained fallback: large closed contours as generic components. Lets the
    full pipeline run before a YOLO model is registered."""

    version = "heuristic-contours:1"

    def predict(self, img: np.ndarray, conf: float) -> List[ObjectComponent]:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)
        contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        tile_area = img.shape[0] * img.shape[1]
        out = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            a = w * h
            if not (0.004 * tile_area <= a <= 0.6 * tile_area) or min(w, h) < 24:
                continue
            fill = cv2.contourArea(cnt) / a
            if fill < 0.15:  # long thin strokes are lines, not components
                continue
            score = float(min(0.99, 0.4 + 0.5 * fill))
            if score >= conf:
                out.append(ObjectComponent(bbox=[x, y, w, h], classification_type="component", confidence=score))
        return out
