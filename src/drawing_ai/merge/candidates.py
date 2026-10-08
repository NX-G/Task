"""Canvas-space view of tile detections used by the Stage 2 patch network."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ..geometry import intersect, seg_box, shift_box, shift_seg
from ..schemas import TileAnnotation

KINDS = ("object", "ocr", "arrow")


@dataclass
class Candidate:
    uid: str  # "<Tile ID>:<kind>:<index>"
    kind: str  # object | ocr | arrow
    tile_id: str
    geom: List[float]  # object/ocr: [x,y,w,h]; arrow: [x1,y1,x2,y2] (canvas px)
    label: str  # class name or OCR text
    confidence: Optional[float] = None
    source_tiles: List[str] = field(default_factory=list)

    @property
    def box(self) -> List[float]:
        return seg_box(self.geom) if self.kind == "arrow" else list(self.geom)

    def to_prompt(self) -> dict:
        d = {"id": self.uid, "kind": self.kind, "geom": [round(v, 1) for v in self.geom], "label": self.label}
        if self.confidence is not None:
            d["confidence"] = round(self.confidence, 3)
        return d


def to_candidates(ann: TileAnnotation) -> List[Candidate]:
    if ann.tile_box is None:
        raise ValueError(f"{ann.tile_id} has no tile_box; cannot place it on the canvas")
    dx, dy = ann.tile_box[0], ann.tile_box[1]
    t = ann.tile_id
    out = []
    for i, o in enumerate(ann.objects):
        out.append(Candidate(f"{t}:object:{i}", "object", t, shift_box(o.bbox, dx, dy), o.classification_type, o.confidence, [t]))
    for i, o in enumerate(ann.ocr):
        out.append(Candidate(f"{t}:ocr:{i}", "ocr", t, shift_box(o.bbox, dx, dy), o.text, o.confidence, [t]))
    for i, a in enumerate(ann.arrows):
        out.append(Candidate(f"{t}:arrow:{i}", "arrow", t, shift_seg(a.points, dx, dy), a.class_id, a.confidence, [t]))
    return out


@dataclass
class TilePair:
    a: str
    b: str
    orientation: str  # "vertical" seam (left/right neighbours) | "horizontal" seam (top/bottom)
    overlap: List[float]  # shared canvas region [x,y,w,h]
    seam: List[float]  # overlap expanded by the seam margin


def adjacent_pairs(anns: List[TileAnnotation], margin: int) -> List[TilePair]:
    """Left/right and top/bottom neighbours (tiles that share a row or column
    and physically overlap). Diagonal neighbours are joined transitively."""
    pairs = []
    boxes: Dict[str, List[int]] = {a.tile_id: a.tile_box for a in anns}
    ids = list(boxes)
    for i, ta in enumerate(ids):
        for tb in ids[i + 1 :]:
            ba, bb = boxes[ta], boxes[tb]
            same_row, same_col = ba[1] == bb[1], ba[0] == bb[0]
            if not (same_row or same_col):
                continue
            ov = _touching_region(ba, bb)
            if ov is None:
                continue
            seam = [ov[0] - margin, ov[1] - margin, ov[2] + 2 * margin, ov[3] + 2 * margin]
            pairs.append(TilePair(ta, tb, "vertical" if same_row else "horizontal", ov, seam))
    return pairs


def _touching_region(a, b) -> Optional[List[float]]:
    inter = intersect(a, b)
    if inter is not None:
        return inter
    # Non-overlapping but abutting tiles (TILE_OVERLAP=0): use the shared edge.
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    if a[1] == b[1] and (ax2 == b[0] or bx2 == a[0]):
        return [min(ax2, bx2), a[1], 0.0, min(a[3], b[3])]
    if a[0] == b[0] and (ay2 == b[1] or by2 == a[1]):
        return [a[0], min(ay2, by2), min(a[2], b[2]), 0.0]
    return None


def in_region(c: Candidate, region: List[float]) -> bool:
    b = c.box
    # inflate degenerate boxes (horizontal / vertical lines) so they intersect
    b = [b[0] - 1, b[1] - 1, b[2] + 2, b[3] + 2]
    return intersect(b, region) is not None


def seam_candidates(cands: Dict[str, List[Candidate]], pair: TilePair) -> Tuple[List[Candidate], List[Candidate]]:
    return (
        [c for c in cands.get(pair.a, []) if in_region(c, pair.seam)],
        [c for c in cands.get(pair.b, []) if in_region(c, pair.seam)],
    )
