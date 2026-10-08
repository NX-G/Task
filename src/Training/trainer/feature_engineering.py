"""Feature engineering: raw drawings + annotations -> tiled training datasets.

Input  (``raw_dir``):
    images/<IMAGE_ID>.{png,jpg,tif,pdf}
    annotations/<IMAGE_ID>.json   either
        - an image-level annotation (spec keys, no "Tile ID", original pixels), or
        - a list of tile annotations (spec format, tile-local canvas pixels).

Every image goes through the *same* preprocessing as inference (orientation,
fixed canvas, tile grid), so Tile IDs and coordinates line up exactly.

Output (``out_dir``):
    tiles/<split>/<IMAGE_ID>_<Tile ID>.png
    annotations/<split>/<IMAGE_ID>_<Tile ID>.json     spec-format tile JSON
    yolo/{images,labels}/<split>/ + data.yaml           Model 1 (components)
    coco_lines/{train,valid,test}/_annotations.coco.json Model 2 (arrows, RF-DETR)
    ocr/<split>.jsonl + ocr/crops/                      Model 3 evaluation
    manifest.json                                       classes, counts, settings
Splits are made per *image* (never per tile) to avoid leakage.
"""

import hashlib
import json
import logging
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import cv2

from drawing_ai.config import Settings
from drawing_ai.geometry import area, clip_segment, intersect, segment_to_box
from drawing_ai.io import SUPPORTED_EXTS, load_pages_from_path
from drawing_ai.preprocessing import Tile, preprocess
from drawing_ai.schemas import AnnotationArrow, ImageAnnotation, ObjectComponent, OCRItem, TileAnnotation

log = logging.getLogger(__name__)
SPLITS = ("train", "valid", "test")


def split_of(image_id: str, ratios=(0.8, 0.1, 0.1)) -> str:
    h = int(hashlib.md5(image_id.encode()).hexdigest(), 16) % 1000 / 1000
    return "train" if h < ratios[0] else "valid" if h < ratios[0] + ratios[1] else "test"


def assign_splits(image_ids: List[str]) -> Dict[str, str]:
    """Hash-based split, but guarantee non-empty valid/test sets (when there are
    at least 3 images) so the trainers always have something to evaluate on."""
    splits = {i: split_of(i) for i in image_ids}
    for need in ("valid", "test"):
        train = sorted(i for i, sp in splits.items() if sp == "train")
        if len(image_ids) >= 3 and need not in splits.values() and len(train) > 1:
            splits[train[-1]] = need
    return splits


def slice_annotation(ann: ImageAnnotation, tf, tiles: List[Tile], min_visible: float = 0.3, min_seg_len: float = 10.0) -> List[TileAnnotation]:
    """Project image-level ground truth onto the canvas and cut it per tile."""
    objs = [(tf.box_to_canvas(o.bbox), o) for o in ann.objects]
    texts = [(tf.box_to_canvas(o.bbox), o) for o in ann.ocr]
    segs = [(tf.seg_to_canvas(a.points), a) for a in ann.arrows]
    out = []
    for t in tiles:
        tb, dx, dy = t.box, t.x, t.y
        ta = TileAnnotation(image_id=ann.image_id, tile_id=t.tile_id, tile_box=tb)
        for box, o in objs:
            vis = intersect(box, tb)
            if vis and area(vis) >= min_visible * area(box):
                ta.objects.append(ObjectComponent(bbox=_loc(vis, dx, dy), classification_type=o.classification_type))
        for box, o in texts:  # text must be (almost) whole, or its label would be wrong
            vis = intersect(box, tb)
            if vis and area(vis) >= 0.9 * area(box):
                ta.ocr.append(OCRItem(bbox=_loc(vis, dx, dy), text=o.text))
        for seg, a in segs:
            c = clip_segment(seg, tb)
            if c and ((c[2] - c[0]) ** 2 + (c[3] - c[1]) ** 2) ** 0.5 >= min_seg_len:
                ta.arrows.append(AnnotationArrow(points=[round(c[0] - dx, 1), round(c[1] - dy, 1), round(c[2] - dx, 1), round(c[3] - dy, 1)], class_id=a.class_id))
        out.append(ta)
    return out


def _loc(b, dx, dy):
    return [round(b[0] - dx, 1), round(b[1] - dy, 1), round(b[2], 1), round(b[3], 1)]


def _load_annotations(path: Path, image_id: str, tf, tiles: List[Tile]) -> List[TileAnnotation]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list) or "Tile ID" in data:
        by_id = {}
        for d in data if isinstance(data, list) else [data]:
            ta = TileAnnotation.model_validate(d)
            by_id[ta.tile_id] = ta
        return [by_id.get(t.tile_id, TileAnnotation(image_id=image_id, tile_id=t.tile_id)).model_copy(update={"tile_box": t.box}) for t in tiles]
    return slice_annotation(ImageAnnotation.model_validate(data), tf, tiles)


def build_datasets(raw_dir: Path, out_dir: Path, s: Settings, empty_ratio: float = 0.2, seed: int = 0) -> dict:
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    rng = random.Random(seed)
    images = sorted(p for p in (raw_dir / "images").iterdir() if p.suffix.lower() in SUPPORTED_EXTS)
    if not images:
        raise FileNotFoundError(f"No images under {raw_dir / 'images'}")

    splits = assign_splits([p.stem for p in images])
    records: Dict[str, List[tuple]] = {k: [] for k in SPLITS}
    obj_classes, arrow_classes = Counter(), Counter()
    for img_path in images:
        image_id = img_path.stem
        ann_path = raw_dir / "annotations" / f"{image_id}.json"
        if not ann_path.exists():
            log.warning("No annotation for %s, skipping", image_id)
            continue
        page = load_pages_from_path(img_path, s.pdf_dpi, 1)[0]
        pre = preprocess(page, s)
        split = splits[image_id]
        for tile, ta in zip(pre.tiles, _load_annotations(ann_path, image_id, pre.transform, pre.tiles)):
            empty = not (ta.objects or ta.arrows or ta.ocr)
            if empty and rng.random() > empty_ratio:
                continue
            name = f"{image_id}_{tile.tile_id}"
            tdir = out_dir / "tiles" / split
            adir = out_dir / "annotations" / split
            tdir.mkdir(parents=True, exist_ok=True)
            adir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(tdir / f"{name}.png"), tile.image)
            (adir / f"{name}.json").write_text(json.dumps(ta.dump(), ensure_ascii=False, indent=1))
            records[split].append((name, tile, ta))
            obj_classes.update(o.classification_type for o in ta.objects)
            arrow_classes.update(a.class_id for a in ta.arrows)

    obj_names = sorted(obj_classes)
    arrow_names = sorted(arrow_classes)
    _write_yolo(out_dir, records, obj_names, s.tile_size)
    _write_coco_lines(out_dir, records, arrow_names)
    _write_ocr(out_dir, records)
    manifest = {
        "settings": {"canvas": [s.canvas_width, s.canvas_height], "tile_size": s.tile_size, "tile_overlap": s.tile_overlap, "pdf_dpi": s.pdf_dpi},
        "object_classes": obj_names,
        "arrow_classes": arrow_names,
        "tiles": {k: len(v) for k, v in records.items()},
        "images": len(images),
        "object_class_counts": dict(obj_classes),
        "arrow_class_counts": dict(arrow_classes),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    log.info("Datasets written to %s: %s", out_dir, manifest["tiles"])
    return manifest


def _write_yolo(out: Path, records, names: List[str], tile_size: int) -> None:
    idx = {n: i for i, n in enumerate(names)}
    for split, recs in records.items():
        idir, ldir = out / "yolo" / "images" / split, out / "yolo" / "labels" / split
        idir.mkdir(parents=True, exist_ok=True)
        ldir.mkdir(parents=True, exist_ok=True)
        for name, tile, ta in recs:
            shutil.copy(out / "tiles" / split / f"{name}.png", idir / f"{name}.png")
            th, tw = tile.image.shape[:2]
            lines = []
            for o in ta.objects:
                x, y, w, h = o.bbox
                lines.append(f"{idx[o.classification_type]} {(x + w / 2) / tw:.6f} {(y + h / 2) / th:.6f} {w / tw:.6f} {h / th:.6f}")
            (ldir / f"{name}.txt").write_text("\n".join(lines))
    (out / "yolo" / "data.yaml").write_text(
        f"path: {(out / 'yolo').resolve()}\ntrain: images/train\nval: images/valid\ntest: images/test\n"
        f"names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(names))
    )
    (out / "yolo" / "classes.json").write_text(json.dumps(names))


def _write_coco_lines(out: Path, records, names: List[str]) -> None:
    """Roboflow-style COCO (category 0 is a placeholder super-category), which
    is the layout RF-DETR expects. Lines are encoded as padded boxes."""
    cats = [{"id": 0, "name": "annotation-lines", "supercategory": "none"}]
    cats += [{"id": i + 1, "name": n, "supercategory": "annotation-lines"} for i, n in enumerate(names)]
    cid = {n: i + 1 for i, n in enumerate(names)}
    for split, recs in records.items():
        d = out / "coco_lines" / split
        d.mkdir(parents=True, exist_ok=True)
        images, anns = [], []
        for img_id, (name, tile, ta) in enumerate(recs):
            shutil.copy(out / "tiles" / split / f"{name}.png", d / f"{name}.png")
            th, tw = tile.image.shape[:2]
            images.append({"id": img_id, "file_name": f"{name}.png", "width": tw, "height": th})
            for a in ta.arrows:
                x, y, w, h = segment_to_box(a.points)
                x, y = max(0.0, x), max(0.0, y)
                w, h = min(w, tw - x), min(h, th - y)
                anns.append({"id": len(anns), "image_id": img_id, "category_id": cid[a.class_id], "bbox": [x, y, w, h], "area": w * h, "iscrowd": 0, "segmentation": []})
        (d / "_annotations.coco.json").write_text(json.dumps({"images": images, "annotations": anns, "categories": cats}))
    (out / "coco_lines" / "classes.json").write_text(json.dumps([c["name"] for c in cats]))


def _write_ocr(out: Path, records) -> None:
    cdir = out / "ocr" / "crops"
    cdir.mkdir(parents=True, exist_ok=True)
    for split, recs in records.items():
        with open(out / "ocr" / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for name, tile, ta in recs:
                for i, o in enumerate(ta.ocr):
                    x, y, w, h = (int(round(v)) for v in o.bbox)
                    crop = tile.image[max(0, y - 3) : y + h + 3, max(0, x - 3) : x + w + 3]
                    if crop.size == 0:
                        continue
                    p = cdir / f"{name}_{i}.png"
                    cv2.imwrite(str(p), crop)
                    f.write(json.dumps({"crop": str(p.relative_to(out)), "text": o.text, "tile": name}, ensure_ascii=False) + "\n")


def load_tile_annotations(dataset_dir: Path, split: Optional[str] = None) -> List[TileAnnotation]:
    root = Path(dataset_dir) / "annotations"
    dirs = [root / split] if split else [root / k for k in SPLITS]
    return [TileAnnotation.model_validate(json.loads(p.read_text())) for d in dirs if d.exists() for p in sorted(d.glob("*.json"))]
