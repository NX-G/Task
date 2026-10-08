"""Document loading: PDF (rendered with PyMuPDF) or raster image -> BGR pages."""

from pathlib import Path
from typing import List

import cv2
import numpy as np

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
SUPPORTED_EXTS = IMAGE_EXTS | {".pdf"}


def load_pages(data: bytes, filename: str, dpi: int = 300, max_pages: int = 10) -> List[np.ndarray]:
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        return _render_pdf(data, dpi, max_pages)
    if ext in IMAGE_EXTS:
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Could not decode image {filename!r}")
        return [img]
    raise ValueError(f"Unsupported file type {ext!r}; expected one of {sorted(SUPPORTED_EXTS)}")


def load_pages_from_path(path: Path, dpi: int = 300, max_pages: int = 10) -> List[np.ndarray]:
    return load_pages(Path(path).read_bytes(), str(path), dpi, max_pages)


def _render_pdf(data: bytes, dpi: int, max_pages: int) -> List[np.ndarray]:
    import fitz  # PyMuPDF

    pages = []
    with fitz.open(stream=data, filetype="pdf") as doc:
        for page in list(doc)[:max_pages]:
            pix = page.get_pixmap(dpi=dpi, alpha=False)
            rgb = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)
            pages.append(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not pages:
        raise ValueError("PDF has no pages")
    return pages


def encode_png(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise ValueError("PNG encoding failed")
    return buf.tobytes()


def decode_image(data: bytes) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image bytes")
    return img
