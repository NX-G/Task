"""Inference pipeline as three idempotent stages keyed by job_id.

    preprocess  : uploaded file -> pages -> orientation / fixed canvas / tiles
    infer       : Stage 1 meta extraction on every tile (3 models)
    postprocess : thresholds -> Stage 2 patch-network merge -> final JSON,
                  BBox visualisation, drift detection

State between stages lives in MongoDB, so each stage can run in the Celery
worker (PIPELINE_EXECUTOR=inline) or as an Airflow task (=airflow).
"""

import logging
import time
from pathlib import Path
from typing import Optional

from . import drift
from .config import Settings, get_settings
from .inference import infer_tiles
from .io import decode_image, encode_png, load_pages
from .models import get_model_suite
from .postprocess import apply_thresholds, build_page_result, merge_tiles, visualize
from .preprocessing import CanvasTransform, divide_into_tiles, preprocess
from .schemas import TileAnnotation
from .storage.mongo import ArtifactStore, get_store

log = logging.getLogger(__name__)


def _store(s: Settings) -> ArtifactStore:
    return get_store(s.mongo_uri, s.mongo_db)


def stage_preprocess(job_id: str, input_file_id: str, filename: str, image_id: Optional[str] = None, s: Settings = None) -> dict:
    s = s or get_settings()
    st = _store(s)
    t0 = time.perf_counter()
    pages = load_pages(st.get_file(input_file_id), filename, s.pdf_dpi, s.max_pages)
    base_id = image_id or Path(filename).stem
    out_pages = []
    for i, page in enumerate(pages):
        pre = preprocess(page, s)
        canvas_id = st.put_file(encode_png(pre.canvas), f"{job_id}_p{i}_canvas.png", "image/png", job_id=job_id, kind="canvas")
        out_pages.append(
            {
                "page": i,
                "image_id": base_id if len(pages) == 1 else f"{base_id}_P{i + 1}",
                "canvas_file_id": canvas_id,
                "transform": pre.transform.to_dict(),
                "tiles": [t.layout() for t in pre.tiles],
            }
        )
    st.init_state(job_id, {"input_file_id": input_file_id, "filename": filename, "pages": out_pages,
                           "timings": {"preprocess_s": round(time.perf_counter() - t0, 3)}})
    return {"pages": len(out_pages), "tiles_per_page": len(out_pages[0]["tiles"])}


def stage_infer(job_id: str, s: Settings = None) -> dict:
    s = s or get_settings()
    st = _store(s)
    state = st.get_state(job_id)
    if not state:
        raise RuntimeError(f"No pipeline state for job {job_id}; run preprocess first")
    t0 = time.perf_counter()
    suite = get_model_suite(s)
    for p in state["pages"]:
        canvas = decode_image(st.get_file(p["canvas_file_id"]))
        tiles = divide_into_tiles(canvas, s.tile_size, s.tile_overlap)
        p["tile_annotations"] = [a.dump() for a in infer_tiles(tiles, p["image_id"], suite, s)]
    timings = {**state.get("timings", {}), "infer_s": round(time.perf_counter() - t0, 3)}
    st.update_state(job_id, {"pages": state["pages"], "model_versions": suite.versions, "timings": timings})
    return {"model_versions": suite.versions}


def stage_postprocess(job_id: str, s: Settings = None) -> dict:
    s = s or get_settings()
    st = _store(s)
    state = st.get_state(job_id)
    if not state or any("tile_annotations" not in p for p in state["pages"]):
        raise RuntimeError(f"Job {job_id} has not been through inference")
    t0 = time.perf_counter()
    baseline = st.latest_baseline() or drift.average_profiles(st.recent_profiles(s.drift_rolling_window))
    pages = []
    for p in state["pages"]:
        tf = CanvasTransform(**p["transform"])
        tiles = apply_thresholds([TileAnnotation.model_validate(a) for a in p["tile_annotations"]], s)
        merged, report = merge_tiles(tiles, s)
        page = build_page_result(p["image_id"], p["page"], tf, tiles, merged, report)

        vis = visualize(decode_image(st.get_file(p["canvas_file_id"])), merged)
        page["visualization_file_id"] = st.put_file(encode_png(vis), f"{job_id}_p{p['page']}_bbox.png", "image/png", job_id=job_id, kind="visualization")

        prof = drift.profile(tiles)
        page["drift"] = drift.compare(baseline, prof, s.drift_psi_threshold)
        st.save_drift({"job_id": job_id, "page": p["page"], "profile": prof, "result": page["drift"], "merged": drift.merged_profile(merged)})
        pages.append(page)

    timings = {**state.get("timings", {}), "postprocess_s": round(time.perf_counter() - t0, 3)}
    st.save_result(job_id, {
        "filename": state["filename"],
        "model_versions": state.get("model_versions", {}),
        "settings": {"canvas": [s.canvas_width, s.canvas_height], "tile_size": s.tile_size, "tile_overlap": s.tile_overlap},
        "timings": timings,
        "pages": pages,
    })
    return {"pages": len(pages), "drift": any(pg["drift"]["drift"] for pg in pages)}


def run_inline(job_id: str, input_file_id: str, filename: str, image_id: Optional[str] = None, s: Settings = None) -> dict:
    s = s or get_settings()
    stage_preprocess(job_id, input_file_id, filename, image_id, s)
    stage_infer(job_id, s)
    return stage_postprocess(job_id, s)
