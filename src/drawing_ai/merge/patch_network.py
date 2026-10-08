"""Stage 2 — Patch Network.

For each pair of neighbouring tiles, the detections near their shared seam are
sent (as JSON) to an LLM which decides which points / boxes / text fragments are
the same entity and what their class / text should be. Decisions from all pairs
are joined with union-find (so an entity spanning 3+ tiles collapses into one)
and every group is fused into a single canvas-space item. The final JSON is the
combined merged output of all tiles.
"""

import logging
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from ..config import Settings
from ..geometry import collinear_merge, intersect, area, union_box
from ..schemas import AnnotationArrow, MergedAnnotation, ObjectComponent, OCRItem, TileAnnotation
from .candidates import Candidate, adjacent_pairs, seam_candidates, to_candidates
from .decision import PairDecision
from .heuristic import decide_pair
from .llm import LLMMergeClient

log = logging.getLogger(__name__)


class _UnionFind:
    def __init__(self, items):
        self.p = {i: i for i in items}

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def _vote(members: List[Candidate]) -> str:
    score: Dict[str, float] = defaultdict(float)
    for m in members:
        score[m.label] += m.confidence if m.confidence is not None else 0.5
    return max(score.items(), key=lambda kv: (kv[1], len(kv[0])))[0]


def _join(a: str, b: str) -> Optional[str]:
    """Join two OCR fragments when one contains the other or they share
    overlapping characters (seen by both tiles); None if unrelated."""
    if not a or b in a:
        return a or b
    if a in b:
        return b
    for k in range(min(len(a), len(b)) - 1, 1, -1):
        if a.endswith(b[:k]):
            return a + b[k:]
    return None


def merge_text(a: str, b: str) -> str:
    joined = _join(a, b)
    return joined if joined is not None else f"{a} {b}"


def _fuse_text(members: List[Candidate]) -> str:
    vertical = all(m.geom[3] > 2 * m.geom[2] for m in members)
    ordered = sorted(members, key=lambda m: (m.geom[1], m.geom[0]) if vertical else (m.geom[0], m.geom[1]))
    text, box = ordered[0].label, ordered[0].geom
    for m in ordered[1:]:
        joined = _join(text, m.label)
        if joined is None:
            inter = intersect(box, m.geom)
            dup = inter is not None and area(inter) >= 0.6 * min(area(box), area(m.geom))
            # same region read differently by two tiles: keep the fuller reading
            joined = max((text, m.label), key=len) if dup else f"{text} {m.label}"
        text = joined
        box = union_box([box, m.geom])
    return text


def _fuse(members: List[Candidate], label: Optional[str]):
    tiles = sorted({t for m in members for t in m.source_tiles})
    confs = [m.confidence for m in members if m.confidence is not None]
    conf = round(max(confs), 4) if confs else None
    kind = members[0].kind
    if kind == "object":
        return ObjectComponent(bbox=_r(union_box([m.geom for m in members])), classification_type=label or _vote(members), confidence=conf, source_tiles=tiles)
    if kind == "arrow":
        pts = collinear_merge([m.geom for m in members]) if len(members) > 1 else members[0].geom
        return AnnotationArrow(points=_r(pts), class_id=label or _vote(members), confidence=conf, source_tiles=tiles)
    text = label or (_fuse_text(members) if len(members) > 1 else members[0].label)
    return OCRItem(bbox=_r(union_box([m.geom for m in members])), text=text, confidence=conf, source_tiles=tiles)


def _r(v):
    return [round(float(x), 1) for x in v]


class PatchNetwork:
    def __init__(self, s: Settings, llm: Optional[LLMMergeClient] = None):
        self.s = s
        self.llm = llm
        if llm is None and s.merge_backend in ("auto", "llm") and s.llm_base_url:
            self.llm = LLMMergeClient(s)
        if s.merge_backend == "llm" and self.llm is None:
            raise ValueError("MERGE_BACKEND=llm requires LLM_BASE_URL")

    @property
    def backend(self) -> str:
        return "llm" if self.llm else "heuristic"

    def _decide(self, pair, a, b, report) -> PairDecision:
        hints = decide_pair(a, b, pair, self.s)
        if self.llm is None:
            return hints
        try:
            report["llm_calls"] += 1
            return self.llm.decide(pair, a, b, hints)
        except Exception as exc:
            report["llm_failures"] += 1
            log.warning("LLM merge failed for %s/%s (%s); using heuristic", pair.a, pair.b, exc)
            hints.source = "heuristic_fallback"
            return hints

    def merge(self, anns: List[TileAnnotation]) -> Tuple[MergedAnnotation, dict]:
        report = {"backend": self.backend, "prompt_version": self.s.prompt_version if self.llm else None,
                  "pairs": 0, "llm_calls": 0, "llm_failures": 0, "merged_groups": 0}
        cands = {a.tile_id: to_candidates(a) for a in anns}
        by_uid = {c.uid: c for cs in cands.values() for c in cs}
        uf = _UnionFind(by_uid)
        labels: Dict[str, str] = {}
        for pair in adjacent_pairs(anns, self.s.merge_seam_margin):
            a, b = seam_candidates(cands, pair)
            if not a or not b:
                continue
            report["pairs"] += 1
            decision = self._decide(pair, a, b, report)
            for m in decision.merges:
                for other in m.ids[1:]:
                    uf.union(m.ids[0], other)
                if m.label:
                    for i in m.ids:
                        labels[i] = m.label
            for uid, lab in decision.relabels.items():
                labels.setdefault(uid, lab)
                by_uid[uid].label = lab

        groups: Dict[str, List[Candidate]] = defaultdict(list)
        for uid in by_uid:
            groups[uf.find(uid)].append(by_uid[uid])

        merged = MergedAnnotation()
        for members in groups.values():
            if len(members) > 1:
                report["merged_groups"] += 1
            label = next((labels[m.uid] for m in members if m.uid in labels and len(members) > 1), None)
            item = _fuse(members, label)
            {"object": merged.objects, "ocr": merged.ocr, "arrow": merged.arrows}[members[0].kind].append(item)
        merged.objects.sort(key=lambda o: (o.bbox[1], o.bbox[0]))
        merged.ocr.sort(key=lambda o: (o.bbox[1], o.bbox[0]))
        merged.arrows.sort(key=lambda o: (o.points[1], o.points[0]))
        return merged, report
