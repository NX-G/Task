"""Small geometry helpers shared by tiling, merging and dataset conversion."""

import math
from typing import List, Optional, Sequence, Tuple

Box = Sequence[float]  # COCO [x, y, w, h]
Seg = Sequence[float]  # [x1, y1, x2, y2]


def xywh_to_xyxy(b: Box) -> Tuple[float, float, float, float]:
    return b[0], b[1], b[0] + b[2], b[1] + b[3]


def xyxy_to_xywh(x1: float, y1: float, x2: float, y2: float) -> List[float]:
    return [x1, y1, x2 - x1, y2 - y1]


def area(b: Box) -> float:
    return max(0.0, b[2]) * max(0.0, b[3])


def intersect(a: Box, b: Box) -> Optional[List[float]]:
    ax1, ay1, ax2, ay2 = xywh_to_xyxy(a)
    bx1, by1, bx2, by2 = xywh_to_xyxy(b)
    x1, y1, x2, y2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
    if x2 <= x1 or y2 <= y1:
        return None
    return xyxy_to_xywh(x1, y1, x2, y2)


def iou(a: Box, b: Box) -> float:
    inter = intersect(a, b)
    if inter is None:
        return 0.0
    i = area(inter)
    return i / (area(a) + area(b) - i + 1e-9)


def union_box(boxes: Sequence[Box]) -> List[float]:
    x1 = min(b[0] for b in boxes)
    y1 = min(b[1] for b in boxes)
    x2 = max(b[0] + b[2] for b in boxes)
    y2 = max(b[1] + b[3] for b in boxes)
    return xyxy_to_xywh(x1, y1, x2, y2)


def shift_box(b: Box, dx: float, dy: float) -> List[float]:
    return [b[0] + dx, b[1] + dy, b[2], b[3]]


def shift_seg(s: Seg, dx: float, dy: float) -> List[float]:
    return [s[0] + dx, s[1] + dy, s[2] + dx, s[3] + dy]


def seg_box(s: Seg) -> List[float]:
    return xyxy_to_xywh(min(s[0], s[2]), min(s[1], s[3]), max(s[0], s[2]), max(s[1], s[3]))


def seg_angle_deg(s: Seg) -> float:
    """Undirected angle in [0, 180)."""
    return math.degrees(math.atan2(s[3] - s[1], s[2] - s[0])) % 180.0


def angle_diff_deg(a: float, b: float) -> float:
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def point_line_distance(px: float, py: float, s: Seg) -> float:
    x1, y1, x2, y2 = s
    dx, dy = x2 - x1, y2 - y1
    n = math.hypot(dx, dy)
    if n < 1e-9:
        return math.hypot(px - x1, py - y1)
    return abs(dy * px - dx * py + x2 * y1 - y2 * x1) / n


def collinear_merge(segs: Sequence[Seg]) -> List[float]:
    """Merge near-collinear segments into the longest spanning segment."""
    ref = max(segs, key=lambda s: math.hypot(s[2] - s[0], s[3] - s[1]))
    ux, uy = ref[2] - ref[0], ref[3] - ref[1]
    n = math.hypot(ux, uy) or 1.0
    ux, uy = ux / n, uy / n
    pts = [(s[0], s[1]) for s in segs] + [(s[2], s[3]) for s in segs]
    proj = [((x - ref[0]) * ux + (y - ref[1]) * uy, x, y) for x, y in pts]
    lo = min(proj)
    hi = max(proj)
    return [lo[1], lo[2], hi[1], hi[2]]


def segment_gap(a: Seg, b: Seg) -> float:
    """Smallest endpoint-to-segment distance (0 if they overlap/touch)."""
    return min(
        _point_seg_dist(a[0], a[1], b),
        _point_seg_dist(a[2], a[3], b),
        _point_seg_dist(b[0], b[1], a),
        _point_seg_dist(b[2], b[3], a),
    )


def _point_seg_dist(px: float, py: float, s: Seg) -> float:
    x1, y1, x2, y2 = s
    dx, dy = x2 - x1, y2 - y1
    l2 = dx * dx + dy * dy
    t = 0.0 if l2 < 1e-9 else max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / l2))
    return math.hypot(px - (x1 + t * dx), py - (y1 + t * dy))


def clip_segment(s: Seg, rect: Box) -> Optional[List[float]]:
    """Liang-Barsky clip of a segment to a COCO rectangle."""
    x1, y1, x2, y2 = s
    xmin, ymin, xmax, ymax = xywh_to_xyxy(rect)
    dx, dy = x2 - x1, y2 - y1
    t0, t1 = 0.0, 1.0
    for p, q in ((-dx, x1 - xmin), (dx, xmax - x1), (-dy, y1 - ymin), (dy, ymax - y1)):
        if abs(p) < 1e-12:
            if q < 0:
                return None
            continue
        t = q / p
        if p < 0:
            t0 = max(t0, t)
        else:
            t1 = min(t1, t)
        if t0 > t1:
            return None
    return [x1 + t0 * dx, y1 + t0 * dy, x1 + t1 * dx, y1 + t1 * dy]


def segment_to_box(s: Seg, pad: float = 4.0) -> List[float]:
    """Encode a line as a detection box (used to train RF-DETR on arrows)."""
    b = seg_box(s)
    return [b[0] - pad, b[1] - pad, b[2] + 2 * pad, b[3] + 2 * pad]


def box_to_segment(b: Box, pad: float = 4.0, diagonal: str = "down") -> List[float]:
    """Decode a detection box back into a line. Thin boxes become horizontal /
    vertical lines through the centre; square-ish boxes become a diagonal whose
    direction ("down" = top-left→bottom-right, "up" = bottom-left→top-right) is
    chosen by the caller from image evidence."""
    x, y, w, h = b
    x1, y1, x2, y2 = x + pad, y + pad, x + w - pad, y + h - pad
    if x2 < x1:
        x1 = x2 = x + w / 2
    if y2 < y1:
        y1 = y2 = y + h / 2
    if w >= 3 * h:
        cy = y + h / 2
        return [x1, cy, x2, cy]
    if h >= 3 * w:
        cx = x + w / 2
        return [cx, y1, cx, y2]
    return [x1, y1, x2, y2] if diagonal == "down" else [x1, y2, x2, y1]
