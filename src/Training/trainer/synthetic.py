"""Synthetic engineering drawings with exact ground truth, for smoke-testing
the full training pipeline before real annotated drawings are available.

Writes ``<out>/images/IMG_xxx.png`` and image-level annotations
``<out>/annotations/IMG_xxx.json`` (original-pixel coordinates).
"""

import json
import math
import random
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

from drawing_ai.schemas import AnnotationArrow, ImageAnnotation, ObjectComponent, OCRItem

W, H = 3508, 2480  # A4 landscape @ 300 dpi
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _arrowhead(img, tip: Tuple[int, int], tail: Tuple[int, int], size=18, thick=2):
    ang = math.atan2(tip[1] - tail[1], tip[0] - tail[0])
    for da in (math.radians(155), -math.radians(155)):
        p = (int(tip[0] + size * math.cos(ang + da)), int(tip[1] + size * math.sin(ang + da)))
        cv2.line(img, tip, p, 0, thick, cv2.LINE_AA)


def _text(img, txt: str, x: int, y: int, scale: float = 1.0) -> OCRItem:
    (tw, th), base = cv2.getTextSize(txt, FONT, scale, 2)
    cv2.putText(img, txt, (x, y), FONT, scale, 0, 2, cv2.LINE_AA)
    return OCRItem(bbox=[x, y - th, tw, th + base], text=txt)


def _overlaps(box, boxes, pad=40):
    x, y, w, h = box
    return any(not (x + w + pad < bx or bx + bw + pad < x or y + h + pad < by or by + bh + pad < y) for bx, by, bw, bh in boxes)


def make_drawing(rng: random.Random, image_id: str) -> Tuple[np.ndarray, ImageAnnotation]:
    img = np.full((H, W), 255, np.uint8)
    cv2.rectangle(img, (60, 60), (W - 60, H - 60), 0, 4)  # sheet frame
    cv2.rectangle(img, (W - 900, H - 300), (W - 60, H - 60), 0, 3)  # title block
    objs: List[ObjectComponent] = []
    ocr: List[OCRItem] = [_text(img, f"DWG {image_id}", W - 860, H - 200, 1.6), _text(img, "SCALE 1:2", W - 860, H - 120, 1.2)]
    arrows: List[AnnotationArrow] = []
    placed: List[List[int]] = []

    for _ in range(rng.randint(6, 12)):
        cls = rng.choice(["Ctype_1", "Ctype_2", "Ctype_3"])
        w, h = rng.randint(180, 620), rng.randint(180, 520)
        if cls == "Ctype_2":
            h = w = min(w, h)
        for _attempt in range(30):
            x, y = rng.randint(200, W - 1200), rng.randint(250, H - 600)
            if not _overlaps([x, y, w, h], placed, 160):
                break
        else:
            continue
        placed.append([x, y, w, h])
        if cls == "Ctype_1":
            cv2.rectangle(img, (x, y), (x + w, y + h), 0, 4)
            cv2.rectangle(img, (x + w // 5, y + h // 5), (x + 4 * w // 5, y + 4 * h // 5), 0, 2)
        elif cls == "Ctype_2":
            c, r = (x + w // 2, y + h // 2), w // 2
            cv2.circle(img, c, r, 0, 4)
            cv2.circle(img, c, r // 3, 0, 3)
            cv2.line(img, (c[0] - r - 20, c[1]), (c[0] + r + 20, c[1]), 0, 1)
            cv2.line(img, (c[0], c[1] - r - 20), (c[0], c[1] + r + 20), 0, 1)
        else:
            pts = np.array([[x + w // 4, y], [x + 3 * w // 4, y], [x + w, y + h // 2], [x + 3 * w // 4, y + h], [x + w // 4, y + h], [x, y + h // 2]])
            cv2.polylines(img, [pts], True, 0, 4)
        objs.append(ObjectComponent(bbox=[x, y, w, h], classification_type=cls))

        # dimension line under the component, with value text
        dy = y + h + rng.randint(50, 90)
        cv2.line(img, (x, y + h + 10), (x, dy + 15), 0, 1)
        cv2.line(img, (x + w, y + h + 10), (x + w, dy + 15), 0, 1)
        cv2.line(img, (x, dy), (x + w, dy), 0, 2)
        _arrowhead(img, (x, dy), (x + w, dy))
        _arrowhead(img, (x + w, dy), (x, dy))
        arrows.append(AnnotationArrow(points=[x, dy, x + w, dy], class_id="dimension_line"))
        ocr.append(_text(img, f"{w / 10:.1f}", x + w // 2 - 40, dy - 12, 1.0))

        # leader line from a note to the component edge
        if rng.random() < 0.7:
            tip = (x + w, y + rng.randint(10, h - 10))
            tail = (tip[0] + rng.randint(120, 320), tip[1] - rng.randint(60, 200))
            cv2.line(img, tip, tail, 0, 2)
            _arrowhead(img, tip, tail)
            arrows.append(AnnotationArrow(points=[*tip, *tail], class_id="leader_line"))
            note = rng.choice(["R12", "M6x1", "4X D8.5", "THRU", "07", "C1.5"])
            ocr.append(_text(img, note, tail[0] + 10, tail[1] + 10, 1.0))

        # section arrow
        if rng.random() < 0.35:
            sx, sy = x - 120, y + h // 2
            cv2.line(img, (sx, sy), (sx + 90, sy), 0, 5)
            _arrowhead(img, (sx + 90, sy), (sx, sy), size=26, thick=5)
            arrows.append(AnnotationArrow(points=[sx, sy, sx + 90, sy], class_id="section_arrow"))
            ocr.append(_text(img, rng.choice(["A", "B", "C"]), sx - 10, sy - 20, 1.4))

    noise = (np.random.default_rng(rng.randint(0, 2**31)).random(img.shape) < 0.0008)
    img[noise] = 0
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), ImageAnnotation(image_id=image_id, objects=objs, ocr=ocr, arrows=arrows)


def generate(out: Path, n: int, seed: int = 7) -> None:
    rng = random.Random(seed)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "annotations").mkdir(parents=True, exist_ok=True)
    for i in range(n):
        image_id = f"IMG_{i + 1:03d}"
        img, ann = make_drawing(rng, image_id)
        cv2.imwrite(str(out / "images" / f"{image_id}.png"), img)
        (out / "annotations" / f"{image_id}.json").write_text(
            json.dumps(ann.model_dump(by_alias=True, exclude_none=True), indent=1, ensure_ascii=False)
        )
