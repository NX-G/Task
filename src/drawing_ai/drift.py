"""Drift detection.

A profile summarises one set of annotations (ground truth at training time, or
predictions at inference time): class distributions per kind and per-tile
detection rates. Inference profiles are compared against the baseline produced
by training post-processing (or, when none exists yet, a rolling baseline of
recent jobs) with the Population Stability Index.
"""

import math
from collections import Counter
from typing import Dict, Iterable, List, Optional

from .schemas import MergedAnnotation, TileAnnotation

EPS = 1e-4


def profile(tiles: Iterable[TileAnnotation]) -> dict:
    obj, arr = Counter(), Counter()
    n_tiles = n_obj = n_arr = n_ocr = 0
    confs: List[float] = []
    for t in tiles:
        n_tiles += 1
        n_obj += len(t.objects)
        n_arr += len(t.arrows)
        n_ocr += len(t.ocr)
        obj.update(o.classification_type for o in t.objects)
        arr.update(a.class_id for a in t.arrows)
        confs += [i.confidence for i in (*t.objects, *t.arrows, *t.ocr) if i.confidence is not None]
    n = max(n_tiles, 1)
    return {
        "n_tiles": n_tiles,
        "object_class_dist": _norm(obj),
        "arrow_class_dist": _norm(arr),
        "objects_per_tile": n_obj / n,
        "arrows_per_tile": n_arr / n,
        "ocr_per_tile": n_ocr / n,
        "mean_confidence": sum(confs) / len(confs) if confs else None,
    }


def merged_profile(m: MergedAnnotation) -> dict:
    return {"objects": len(m.objects), "arrows": len(m.arrows), "ocr": len(m.ocr)}


def _norm(c: Counter) -> Dict[str, float]:
    total = sum(c.values())
    return {k: v / total for k, v in c.items()} if total else {}


def psi(expected: Dict[str, float], actual: Dict[str, float]) -> float:
    keys = set(expected) | set(actual)
    if not keys:
        return 0.0
    total = 0.0
    for k in keys:
        e, a = max(expected.get(k, 0.0), EPS), max(actual.get(k, 0.0), EPS)
        total += (a - e) * math.log(a / e)
    return total


def _rate_shift(expected: float, actual: float) -> float:
    """Symmetric relative change in a per-tile rate, in [0, 2]."""
    return abs(actual - expected) / max((abs(actual) + abs(expected)) / 2, EPS) if (actual or expected) else 0.0


def compare(baseline: Optional[dict], current: dict, psi_threshold: float = 0.25) -> dict:
    if not baseline:
        return {"status": "no_baseline", "drift": False, "metrics": {}, "flags": []}
    metrics = {
        "object_class_psi": psi(baseline.get("object_class_dist", {}), current["object_class_dist"]),
        "arrow_class_psi": psi(baseline.get("arrow_class_dist", {}), current["arrow_class_dist"]),
        "objects_per_tile_shift": _rate_shift(baseline.get("objects_per_tile", 0), current["objects_per_tile"]),
        "arrows_per_tile_shift": _rate_shift(baseline.get("arrows_per_tile", 0), current["arrows_per_tile"]),
        "ocr_per_tile_shift": _rate_shift(baseline.get("ocr_per_tile", 0), current["ocr_per_tile"]),
    }
    if baseline.get("mean_confidence") and current.get("mean_confidence"):
        metrics["confidence_drop"] = max(0.0, baseline["mean_confidence"] - current["mean_confidence"])
    limits = {"_psi": psi_threshold, "_shift": 1.0, "confidence_drop": 0.15}
    flags = [k for k, v in metrics.items() if any(k.endswith(sfx) and v > lim for sfx, lim in limits.items())]
    return {"status": "ok", "drift": bool(flags), "metrics": {k: round(v, 4) for k, v in metrics.items()}, "flags": flags}


def average_profiles(profiles: List[dict]) -> Optional[dict]:
    """Rolling baseline from previous inference profiles."""
    profiles = [p for p in profiles if p and p.get("n_tiles")]
    if not profiles:
        return None
    out = {"n_tiles": sum(p["n_tiles"] for p in profiles)}
    for key in ("object_class_dist", "arrow_class_dist"):
        acc: Counter = Counter()
        for p in profiles:
            acc.update(p.get(key, {}))
        out[key] = {k: v / len(profiles) for k, v in acc.items()}
    for key in ("objects_per_tile", "arrows_per_tile", "ocr_per_tile"):
        out[key] = sum(p[key] for p in profiles) / len(profiles)
    confs = [p["mean_confidence"] for p in profiles if p.get("mean_confidence") is not None]
    out["mean_confidence"] = sum(confs) / len(confs) if confs else None
    return out
