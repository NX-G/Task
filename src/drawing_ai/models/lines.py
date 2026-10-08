"""Model 2 — annotation line / arrow detection (RF-DETR).

RF-DETR is a box detector, so each annotation line is trained as a padded box
around the segment (geometry.segment_to_box) and decoded back to a segment at
inference (geometry.box_to_segment). For square-ish boxes the diagonal
direction is picked by measuring ink along both diagonals.
"""

import json
import math
from pathlib import Path
from typing import List

import cv2
import numpy as np

from ..geometry import box_to_segment
from ..schemas import AnnotationArrow


def _ink_along(gray: np.ndarray, x1: float, y1: float, x2: float, y2: float) -> float:
    n = int(max(abs(x2 - x1), abs(y2 - y1))) + 1
    xs = np.clip(np.linspace(x1, x2, n).astype(int), 0, gray.shape[1] - 1)
    ys = np.clip(np.linspace(y1, y2, n).astype(int), 0, gray.shape[0] - 1)
    return float((255 - gray[ys, xs]).mean())


class RFDETRLineDetector:
    def __init__(self, model_dir: Path, version: str, device: str = "cpu"):
        from rfdetr import RFDETRBase

        model_dir = Path(model_dir)
        self.classes: List[str] = json.loads((model_dir / "classes.json").read_text())
        self.model = RFDETRBase(pretrain_weights=str(model_dir / "checkpoint_best_total.pth"), num_classes=len(self.classes), device=device)
        self.version = f"rfdetr:{version}"

    def predict(self, img: np.ndarray, conf: float) -> List[AnnotationArrow]:
        from PIL import Image

        rgb = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        det = self.model.predict(rgb, threshold=conf)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        out = []
        for (x1, y1, x2, y2), cid, p in zip(det.xyxy.tolist(), det.class_id.tolist(), det.confidence.tolist()):
            box = [x1, y1, x2 - x1, y2 - y1]
            down = _ink_along(gray, x1, y1, x2, y2)
            up = _ink_along(gray, x1, y2, x2, y1)
            seg = box_to_segment(box, diagonal="down" if down >= up else "up")
            name = self.classes[int(cid)] if 0 <= int(cid) < len(self.classes) else str(cid)
            out.append(AnnotationArrow(points=[round(v, 1) for v in seg], class_id=name, confidence=float(p)))
        return out


class HoughLineDetector:
    """Untrained fallback: probabilistic Hough transform on thin strokes."""

    version = "heuristic-hough:1"

    def __init__(self, min_len: int = 60, max_lines: int = 150):
        self.min_len = min_len
        self.max_lines = max_lines

    def predict(self, img: np.ndarray, conf: float) -> List[AnnotationArrow]:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # keep thin strokes: remove thick filled regions (opening detects them)
        thick = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        thin = cv2.subtract(bw, thick)
        lines = cv2.HoughLinesP(thin, 1, np.pi / 180, threshold=60, minLineLength=self.min_len, maxLineGap=4)
        if lines is None:
            return []
        segs = sorted((l[0].tolist() for l in lines), key=lambda s: -math.hypot(s[2] - s[0], s[3] - s[1]))
        diag = math.hypot(*img.shape[:2])
        out = []
        for x1, y1, x2, y2 in segs[: self.max_lines]:
            score = min(0.95, 0.3 + math.hypot(x2 - x1, y2 - y1) / diag)
            if score >= conf:
                out.append(AnnotationArrow(points=[x1, y1, x2, y2], class_id="line_candidate", confidence=round(score, 3)))
        return out
