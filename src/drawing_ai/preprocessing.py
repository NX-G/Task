"""Preprocessing: orientation, fixed-dimension canvas, enhancement, tile division.

Every drawing is mapped onto one fixed canvas (CANVAS_WIDTH x CANVAS_HEIGHT) so
all images share the same dimensions and the same tile grid. Tiles overlap by
TILE_OVERLAP pixels so objects cut by a seam are seen whole by at least one tile
and can be stitched back by the Stage 2 patch network.
"""

from dataclasses import asdict, dataclass
from typing import List, Sequence, Tuple

import cv2
import numpy as np

from .config import Settings


@dataclass
class CanvasTransform:
    """Maps original-image pixels <-> canvas pixels.

    original --(rotate 90° CW if portrait)--> rotated --(scale, pad)--> canvas
    """

    orig_w: int
    orig_h: int
    rotated: bool
    scale: float
    pad_x: int
    pad_y: int
    canvas_w: int
    canvas_h: int

    def to_canvas(self, x: float, y: float) -> Tuple[float, float]:
        if self.rotated:
            x, y = self.orig_h - 1 - y, x
        return x * self.scale + self.pad_x, y * self.scale + self.pad_y

    def to_original(self, x: float, y: float) -> Tuple[float, float]:
        x, y = (x - self.pad_x) / self.scale, (y - self.pad_y) / self.scale
        if self.rotated:
            x, y = y, self.orig_h - 1 - x
        return x, y

    def box_to_canvas(self, b: Sequence[float]) -> List[float]:
        xs, ys = zip(*(self.to_canvas(px, py) for px, py in _corners(b)))
        return [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]

    def box_to_original(self, b: Sequence[float]) -> List[float]:
        xs, ys = zip(*(self.to_original(px, py) for px, py in _corners(b)))
        return [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]

    def seg_to_canvas(self, s: Sequence[float]) -> List[float]:
        return [*self.to_canvas(s[0], s[1]), *self.to_canvas(s[2], s[3])]

    def seg_to_original(self, s: Sequence[float]) -> List[float]:
        return [*self.to_original(s[0], s[1]), *self.to_original(s[2], s[3])]

    def to_dict(self) -> dict:
        return asdict(self)


def _corners(b: Sequence[float]):
    x, y, w, h = b
    return [(x, y), (x + w, y), (x, y + h), (x + w, y + h)]


@dataclass
class Tile:
    tile_id: str
    index: int
    row: int
    col: int
    x: int
    y: int
    w: int
    h: int
    image: np.ndarray

    @property
    def box(self) -> List[int]:
        return [self.x, self.y, self.w, self.h]

    def layout(self) -> dict:
        return {"tile_id": self.tile_id, "index": self.index, "row": self.row, "col": self.col, "box": self.box}


@dataclass
class PreprocessedImage:
    canvas: np.ndarray
    transform: CanvasTransform
    tiles: List[Tile]


def normalize_orientation(img: np.ndarray) -> Tuple[np.ndarray, bool]:
    """Engineering sheets are landscape; rotate portrait scans 90° clockwise."""
    h, w = img.shape[:2]
    if h > w:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE), True
    return img, False


def enhance(img: np.ndarray) -> np.ndarray:
    """Grayscale + min-max contrast stretch, returned as 3-channel BGR so every
    Stage 1 model receives the same input format. Deliberately no blurring:
    1-px dimension lines must survive."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def to_fixed_canvas(img: np.ndarray, canvas_w: int, canvas_h: int, rotated: bool, orig_w: int, orig_h: int):
    h, w = img.shape[:2]
    scale = min(canvas_w / w, canvas_h / h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    resized = cv2.resize(img, (nw, nh), interpolation=interp)
    pad_x, pad_y = (canvas_w - nw) // 2, (canvas_h - nh) // 2
    canvas = np.full((canvas_h, canvas_w, 3), 255, np.uint8)
    canvas[pad_y : pad_y + nh, pad_x : pad_x + nw] = resized
    tf = CanvasTransform(orig_w, orig_h, rotated, scale, pad_x, pad_y, canvas_w, canvas_h)
    return canvas, tf


def tile_starts(length: int, tile: int, overlap: int) -> List[int]:
    if length <= tile:
        return [0]
    stride = tile - overlap
    starts = list(range(0, length - tile, stride))
    starts.append(length - tile)  # last tile flush with the edge
    return sorted(set(starts))


def tile_grid(canvas_w: int, canvas_h: int, tile: int, overlap: int) -> List[dict]:
    """Deterministic row-major grid: Tile_001 is top-left."""
    layout = []
    for r, y in enumerate(tile_starts(canvas_h, tile, overlap)):
        for c, x in enumerate(tile_starts(canvas_w, tile, overlap)):
            idx = len(layout)
            layout.append(
                {
                    "tile_id": f"Tile_{idx + 1:03d}",
                    "index": idx,
                    "row": r,
                    "col": c,
                    "box": [x, y, min(tile, canvas_w - x), min(tile, canvas_h - y)],
                }
            )
    return layout


def divide_into_tiles(canvas: np.ndarray, tile: int, overlap: int) -> List[Tile]:
    h, w = canvas.shape[:2]
    tiles = []
    for t in tile_grid(w, h, tile, overlap):
        x, y, tw, th = t["box"]
        tiles.append(Tile(t["tile_id"], t["index"], t["row"], t["col"], x, y, tw, th, canvas[y : y + th, x : x + tw].copy()))
    return tiles


def preprocess(img: np.ndarray, settings: Settings) -> PreprocessedImage:
    orig_h, orig_w = img.shape[:2]
    oriented, rotated = normalize_orientation(img)
    canvas, tf = to_fixed_canvas(oriented, settings.canvas_width, settings.canvas_height, rotated, orig_w, orig_h)
    canvas = enhance(canvas)
    return PreprocessedImage(canvas, tf, divide_into_tiles(canvas, settings.tile_size, settings.tile_overlap))
