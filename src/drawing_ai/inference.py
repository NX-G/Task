"""Stage 1 — meta extraction. Every tile is passed through the three models
(component detector, annotation-line detector, OCR) independently."""

import logging
from typing import List

from .config import Settings
from .models import ModelSuite
from .preprocessing import Tile
from .schemas import TileAnnotation

log = logging.getLogger(__name__)


def infer_tile(tile: Tile, image_id: str, suite: ModelSuite, s: Settings) -> TileAnnotation:
    return TileAnnotation(
        image_id=image_id,
        tile_id=tile.tile_id,
        objects=suite.detector.predict(tile.image, s.detector_conf),
        arrows=suite.line_detector.predict(tile.image, s.line_conf),
        ocr=suite.ocr.predict(tile.image, s.ocr_conf),
        tile_box=tile.box,
    )


def infer_tiles(tiles: List[Tile], image_id: str, suite: ModelSuite, s: Settings) -> List[TileAnnotation]:
    out = []
    for t in tiles:
        ann = infer_tile(t, image_id, suite, s)
        log.debug("%s %s: %d obj, %d arrows, %d ocr", image_id, t.tile_id, len(ann.objects), len(ann.arrows), len(ann.ocr))
        out.append(ann)
    return out
