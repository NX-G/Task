"""Deterministic merge rules for one tile pair. Used directly when no LLM is
configured, as hints in the LLM prompt, and as the fallback if the LLM fails."""

from typing import List

from ..config import Settings
from ..geometry import angle_diff_deg, intersect, iou, point_line_distance, seg_angle_deg, segment_gap
from .candidates import Candidate, TilePair
from .decision import Merge, PairDecision


def _clipped_iou(a: List[float], b: List[float], region: List[float]) -> float:
    # Pad the overlap region so zero-width seams (no tile overlap) still work.
    region = [region[0] - 2, region[1] - 2, region[2] + 4, region[3] + 4]
    ca, cb = intersect(a, region), intersect(b, region)
    if ca is None or cb is None:
        return 0.0
    return iou(ca, cb)


def same_object(a: Candidate, b: Candidate, pair: TilePair, s: Settings) -> bool:
    return _clipped_iou(a.geom, b.geom, pair.overlap) >= s.merge_overlap_iou or iou(a.geom, b.geom) >= s.merge_overlap_iou


def same_arrow(a: Candidate, b: Candidate, s: Settings) -> bool:
    if angle_diff_deg(seg_angle_deg(a.geom), seg_angle_deg(b.geom)) > s.merge_angle_tol_deg:
        return False
    (x1, y1, x2, y2), ref = b.geom, a.geom
    if max(point_line_distance(x1, y1, ref), point_line_distance(x2, y2, ref)) > s.merge_line_dist_tol:
        return False
    return segment_gap(a.geom, b.geom) <= s.merge_endpoint_tol


def same_text(a: Candidate, b: Candidate, pair: TilePair, s: Settings) -> bool:
    if _clipped_iou(a.geom, b.geom, pair.overlap) >= s.merge_overlap_iou:
        return True
    ax, ay, aw, ah = a.geom
    bx, by, bw, bh = b.geom
    same_baseline = abs((ay + ah / 2) - (by + bh / 2)) <= 0.5 * min(ah, bh)
    similar_height = min(ah, bh) / max(ah, bh, 1e-6) >= 0.6
    gap = max(bx - (ax + aw), ax - (bx + bw))
    return pair.orientation == "vertical" and same_baseline and similar_height and gap <= s.merge_endpoint_tol


def decide_pair(a_items: List[Candidate], b_items: List[Candidate], pair: TilePair, s: Settings) -> PairDecision:
    merges = []
    for a in a_items:
        for b in b_items:
            if a.kind != b.kind:
                continue
            if a.kind == "object":
                ok = same_object(a, b, pair, s)
            elif a.kind == "arrow":
                ok = same_arrow(a, b, s)
            else:
                ok = same_text(a, b, pair, s)
            if ok:
                merges.append(Merge([a.uid, b.uid]))
    return PairDecision(merges=merges, source="heuristic")
