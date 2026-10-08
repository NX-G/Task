"""Post-processing: thresholding, tile merging, BBox visualisation, final JSON."""

from typing import List, Tuple

import cv2
import numpy as np

from .config import Settings
from .merge import PatchNetwork
from .preprocessing import CanvasTransform
from .schemas import MergedAnnotation, TileAnnotation

COLORS = {"object": (40, 160, 40), "ocr": (200, 90, 0), "arrow": (30, 30, 220)}  # BGR


def apply_thresholds(anns: List[TileAnnotation], s: Settings) -> List[TileAnnotation]:
    def keep(items, thr):
        return [i for i in items if i.confidence is None or i.confidence >= thr]

    return [
        a.model_copy(update={"objects": keep(a.objects, s.detector_conf), "arrows": keep(a.arrows, s.line_conf), "ocr": keep(a.ocr, s.ocr_conf)})
        for a in anns
    ]


def merge_tiles(anns: List[TileAnnotation], s: Settings, network: PatchNetwork = None) -> Tuple[MergedAnnotation, dict]:
    return (network or PatchNetwork(s)).merge(anns)


def to_original_coords(merged: MergedAnnotation, tf: CanvasTransform) -> dict:
    """Same merged JSON expressed in the uploaded image's pixel space."""
    m = merged.model_copy(deep=True)
    for o in m.objects:
        o.bbox = [round(v, 1) for v in tf.box_to_original(o.bbox)]
    for o in m.ocr:
        o.bbox = [round(v, 1) for v in tf.box_to_original(o.bbox)]
    for a in m.arrows:
        a.points = [round(v, 1) for v in tf.seg_to_original(a.points)]
    return m.dump()


def visualize(canvas: np.ndarray, merged: MergedAnnotation, max_side: int = 4096) -> np.ndarray:
    vis = canvas.copy()
    for o in merged.objects:
        x, y, w, h = map(int, o.bbox)
        cv2.rectangle(vis, (x, y), (x + w, y + h), COLORS["object"], 3)
        _label(vis, f"{o.classification_type} {o.confidence or 0:.2f}", x, y, COLORS["object"])
    for o in merged.ocr:
        x, y, w, h = map(int, o.bbox)
        cv2.rectangle(vis, (x, y), (x + w, y + h), COLORS["ocr"], 2)
        _label(vis, o.text[:40], x, y + h + 18, COLORS["ocr"])
    for a in merged.arrows:
        x1, y1, x2, y2 = map(int, a.points)
        cv2.line(vis, (x1, y1), (x2, y2), COLORS["arrow"], 3)
        cv2.circle(vis, (x1, y1), 6, COLORS["arrow"], -1)
    h, w = vis.shape[:2]
    if max(h, w) > max_side:
        f = max_side / max(h, w)
        vis = cv2.resize(vis, (int(w * f), int(h * f)), interpolation=cv2.INTER_AREA)
    return vis


def _label(img, text, x, y, color):
    # cv2 Hershey fonts are ASCII-only; replace common drawing symbols.
    text = text.replace("Ø", "D").replace("±", "+-").replace("°", "deg").encode("ascii", "replace").decode()
    cv2.putText(img, text, (x, max(14, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)


def build_page_result(image_id: str, page: int, tf: CanvasTransform, tiles: List[TileAnnotation], merged: MergedAnnotation, merge_report: dict) -> dict:
    """Final JSON for one page: the per-tile JSON (spec format, tile-local
    coordinates) plus the combined merged JSON (canvas and original coords)."""
    return {
        "page": page,
        "Image ID": image_id,
        "canvas": tf.to_dict(),
        "tiles": [t.dump() for t in tiles],
        "merged": merged.dump(),
        "merged_original_coords": to_original_coords(merged, tf),
        "merge_report": merge_report,
        "counts": {"OBJECT/COMPONENT": len(merged.objects), "OCR": len(merged.ocr), "Annotation Arrow": len(merged.arrows)},
    }
