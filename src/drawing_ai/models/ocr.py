"""Model 3 — OCR (GOT-OCR2.0).

GOT-OCR2.0 reads text but does not localise it, so text regions are proposed
first (Tesseract word boxes when available, otherwise a morphological region
detector) and GOT-OCR2.0 transcribes each crop. The Tesseract backend is the
lightweight fallback that both localises and reads.
"""

import logging
from typing import List, Tuple

import cv2
import numpy as np

from ..schemas import OCRItem

log = logging.getLogger(__name__)

Region = Tuple[int, int, int, int]


def detect_text_regions(img: np.ndarray, min_h: int = 6, max_h: int = 90) -> List[Region]:
    """Morphological text-line proposals: characters dilated horizontally into
    word blobs; long thin strokes (lines) and big blobs (shapes) rejected."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # remove long horizontal / vertical rules so they don't glue text together
    for k in ((40, 1), (1, 40)):
        rules = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, k))
        bw = cv2.subtract(bw, rules)
    blobs = cv2.dilate(bw, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3)))
    contours, _ = cv2.findContours(blobs, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regions = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if not (min_h <= h <= max_h) or w < 6 or w > 25 * h:
            continue
        density = cv2.countNonZero(bw[y : y + h, x : x + w]) / float(w * h)
        if 0.08 <= density <= 0.7:
            regions.append((x, y, w, h))
    return regions


class TesseractOCR:
    version = "tesseract:5"

    def __init__(self):
        import pytesseract

        self.tess = pytesseract
        self.tess.get_tesseract_version()  # fail fast if the binary is missing

    def words(self, img: np.ndarray) -> List[OCRItem]:
        data = self.tess.image_to_data(img, config="--psm 11", output_type=self.tess.Output.DICT)
        lines = {}
        for i, txt in enumerate(data["text"]):
            txt = (txt or "").strip()
            conf = float(data["conf"][i])
            if not txt or conf < 0:
                continue
            key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
            lines.setdefault(key, []).append((data["left"][i], data["top"][i], data["width"][i], data["height"][i], txt, conf))
        out = []
        for words in lines.values():
            words.sort()
            x1 = min(w[0] for w in words)
            y1 = min(w[1] for w in words)
            x2 = max(w[0] + w[2] for w in words)
            y2 = max(w[1] + w[3] for w in words)
            text = " ".join(w[4] for w in words)
            conf = sum(w[5] for w in words) / len(words) / 100.0
            out.append(OCRItem(bbox=[x1, y1, x2 - x1, y2 - y1], text=text, confidence=round(conf, 3)))
        return out

    def predict(self, img: np.ndarray, conf: float) -> List[OCRItem]:
        return [o for o in self.words(img) if (o.confidence or 0) >= conf and any(ch.isalnum() for ch in o.text)]


class GotOCR2:
    """stepfun-ai/GOT-OCR-2.0-hf via transformers (>= 4.49)."""

    def __init__(self, model_id: str, device: str = "cpu", batch_size: int = 8):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(model_id)
        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        self.model = AutoModelForImageTextToText.from_pretrained(model_id, torch_dtype=dtype).to(device).eval()
        self.device = device
        self.batch_size = batch_size
        self.version = f"got-ocr2:{model_id}"
        try:
            self.proposer = TesseractOCR()
        except Exception:
            self.proposer = None

    def _regions(self, img: np.ndarray) -> List[Region]:
        if self.proposer is not None:
            return [tuple(int(v) for v in o.bbox) for o in self.proposer.words(img)]
        return detect_text_regions(img)

    def predict(self, img: np.ndarray, conf: float) -> List[OCRItem]:
        from PIL import Image

        h, w = img.shape[:2]
        regions = []
        for x, y, rw, rh in self._regions(img):
            p = 3  # small margin around each crop
            regions.append((max(0, x - p), max(0, y - p), min(w, x + rw + p), min(h, y + rh + p)))
        out = []
        for i in range(0, len(regions), self.batch_size):
            batch = regions[i : i + self.batch_size]
            crops = [Image.fromarray(cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)) for x1, y1, x2, y2 in batch]
            inputs = self.processor(crops, return_tensors="pt").to(self.device)
            with self.torch.inference_mode():
                ids = self.model.generate(
                    **inputs, do_sample=False, tokenizer=self.processor.tokenizer, stop_strings="<|im_end|>", max_new_tokens=64
                )
            texts = self.processor.batch_decode(ids[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
            for (x1, y1, x2, y2), text in zip(batch, texts):
                text = text.strip()
                if text and any(ch.isalnum() for ch in text):
                    # GOT does not expose a calibrated score; use a fixed prior.
                    out.append(OCRItem(bbox=[x1, y1, x2 - x1, y2 - y1], text=text, confidence=0.9))
        return [o for o in out if (o.confidence or 0) >= conf]


class NoOCR:
    version = "none"

    def predict(self, img: np.ndarray, conf: float) -> List[OCRItem]:
        return []
