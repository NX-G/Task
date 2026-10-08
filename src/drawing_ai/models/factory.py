"""Builds the Stage 1 model suite once per process, honouring the *_BACKEND
settings. ``auto`` prefers the trained model and degrades gracefully."""

import logging
import threading
from dataclasses import dataclass
from typing import Any, Dict, Optional

from ..config import Settings
from .detector import HeuristicDetector, YoloDetector
from .lines import HoughLineDetector, RFDETRLineDetector
from .ocr import GotOCR2, NoOCR, TesseractOCR
from .registry import resolve_model

log = logging.getLogger(__name__)


@dataclass
class ModelSuite:
    detector: Any
    line_detector: Any
    ocr: Any

    @property
    def versions(self) -> Dict[str, str]:
        return {"detector": self.detector.version, "line_detector": self.line_detector.version, "ocr": self.ocr.version}


def _build(kind: str, backend: str, real, fallback):
    if backend in ("auto", kind):
        try:
            model = real()
            if model is not None:
                return model
            if backend == kind:
                raise RuntimeError(f"{kind} requested but no model is registered")
        except Exception as exc:
            if backend == kind:
                raise
            log.warning("%s unavailable (%s: %s); using fallback", kind, type(exc).__name__, exc)
    return fallback()


def build_model_suite(s: Settings) -> ModelSuite:
    def yolo():
        m = resolve_model(s.mlflow_tracking_uri, s.yolo_model_name, s.model_alias, s.model_cache_dir)
        return YoloDetector(m.local_dir, m.version, s.device) if m else None

    def rfdetr():
        m = resolve_model(s.mlflow_tracking_uri, s.rfdetr_model_name, s.model_alias, s.model_cache_dir)
        return RFDETRLineDetector(m.local_dir, m.version, s.device) if m else None

    def got():
        return GotOCR2(s.got_ocr_model_id, s.device)

    def tesseract_or_none():
        try:
            return TesseractOCR()
        except Exception as exc:
            log.warning("tesseract unavailable (%s); OCR disabled", exc)
            return NoOCR()

    detector = _build("yolo", s.detector_backend, yolo, HeuristicDetector)
    lines = _build("rfdetr", s.line_backend, rfdetr, HoughLineDetector)
    if s.ocr_backend == "none":
        ocr = NoOCR()
    elif s.ocr_backend == "tesseract":
        ocr = TesseractOCR()
    elif s.ocr_backend == "auto":
        # GOT-OCR2.0 is only attempted in auto mode when torch/transformers exist.
        try:
            import transformers  # noqa: F401

            ocr = _build("got_ocr2", "auto", got, tesseract_or_none)
        except ImportError:
            ocr = tesseract_or_none()
    else:
        ocr = _build("got_ocr2", s.ocr_backend, got, tesseract_or_none)
    suite = ModelSuite(detector, lines, ocr)
    log.info("Model suite: %s", suite.versions)
    return suite


_suite: Optional[ModelSuite] = None
_lock = threading.Lock()


def get_model_suite(s: Settings, reload: bool = False) -> ModelSuite:
    global _suite
    with _lock:
        if _suite is None or reload:
            _suite = build_model_suite(s)
        return _suite
