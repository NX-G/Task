import json

from drawing_ai import drift
from drawing_ai.config import Settings
from drawing_ai.schemas import ObjectComponent, TileAnnotation
from trainer.feature_engineering import build_datasets, load_tile_annotations
from trainer.synthetic import generate


def test_synthetic_to_datasets(tmp_path):
    s = Settings(canvas_width=2048, canvas_height=1448, tile_size=512, tile_overlap=64)
    generate(tmp_path / "raw", 4)
    m = build_datasets(tmp_path / "raw", tmp_path / "ds", s, empty_ratio=0.0)
    assert set(m["object_classes"]) <= {"Ctype_1", "Ctype_2", "Ctype_3"}
    assert "dimension_line" in m["arrow_classes"]
    assert m["tiles"]["valid"] > 0 and m["tiles"]["test"] > 0

    # YOLO labels are normalised and reference valid class indexes
    for lbl in (tmp_path / "ds/yolo/labels/train").glob("*.txt"):
        for line in filter(None, lbl.read_text().splitlines()):
            c, *xywh = line.split()
            assert 0 <= int(c) < len(m["object_classes"])
            assert all(0.0 <= float(v) <= 1.0 for v in xywh)

    # RF-DETR COCO: placeholder category 0 + one per arrow class
    coco = json.loads((tmp_path / "ds/coco_lines/train/_annotations.coco.json").read_text())
    assert [c["id"] for c in coco["categories"]] == list(range(len(m["arrow_classes"]) + 1))
    assert all(a["category_id"] >= 1 for a in coco["annotations"])

    # spec-format tile JSON round-trips
    tiles = load_tile_annotations(tmp_path / "ds", "train")
    assert tiles and all(t.tile_id.startswith("Tile_") for t in tiles)


def _tile(classes):
    return TileAnnotation(image_id="I", tile_id="Tile_001",
                          objects=[ObjectComponent(bbox=[0, 0, 1, 1], classification_type=c, confidence=0.9) for c in classes])


def test_drift_flags_class_shift():
    base = drift.profile([_tile(["A", "B"])] * 10)
    same = drift.profile([_tile(["A", "B"])] * 5)
    shifted = drift.profile([_tile(["C", "C", "C", "C"])] * 5)
    assert drift.compare(base, same)["drift"] is False
    res = drift.compare(base, shifted)
    assert res["drift"] is True and "object_class_psi" in res["flags"]
    assert drift.compare(None, same)["status"] == "no_baseline"
