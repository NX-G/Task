import json

import pytest

from drawing_ai import geometry as g
from drawing_ai.schemas import TileAnnotation

SPEC_SAMPLE = {
    "Image ID": "IMG_001",
    "Tile ID": "Tile_001",
    "OBJECT/COMPONENT": [
        {"object_Bounding_Box": [120, 340, 85, 42], "classification_type": "Ctype_1"},
        {"object_Bounding_Box": [280, 150, 60, 60], "classification_type": "Ctype_2"},
        {"object_Bounding_Box": [410, 220, 30, 30], "classification_type": "Ctype_3"},
    ],
    "OCR": [
        {"content_bbox": [122, 342, 80, 38], "text_extracted": "Ø25.4"},
        {"content_bbox": [295, 165, 30, 30], "text_extracted": "07"},
    ],
    "Annotation Arrow": [
        {"Points": [205, 361, 280, 361], "class_id": "leader_line"},
        {"Points": [410, 235, 500, 235], "class_id": "section_arrow"},
    ],
}


def test_spec_sample_round_trips():
    ann = TileAnnotation.model_validate(SPEC_SAMPLE)
    assert ann.objects[0].classification_type == "Ctype_1"
    assert ann.ocr[0].text == "Ø25.4"
    assert ann.arrows[1].class_id == "section_arrow"
    assert json.loads(json.dumps(ann.dump(), ensure_ascii=False)) == SPEC_SAMPLE


def test_bbox_must_have_four_values():
    bad = {**SPEC_SAMPLE, "OBJECT/COMPONENT": [{"object_Bounding_Box": [1, 2, 3], "classification_type": "x"}]}
    with pytest.raises(Exception):
        TileAnnotation.model_validate(bad)


def test_iou_and_union():
    assert g.iou([0, 0, 10, 10], [0, 0, 10, 10]) == pytest.approx(1.0)
    assert g.iou([0, 0, 10, 10], [20, 20, 5, 5]) == 0.0
    assert g.union_box([[0, 0, 10, 10], [5, 5, 10, 10]]) == [0, 0, 15, 15]


def test_clip_segment():
    assert g.clip_segment([-10, 5, 30, 5], [0, 0, 20, 20]) == pytest.approx([0, 5, 20, 5])
    assert g.clip_segment([30, 30, 40, 40], [0, 0, 20, 20]) is None


def test_collinear_merge_spans_both():
    assert g.collinear_merge([[0, 0, 50, 0], [40, 0, 100, 0]]) == pytest.approx([0, 0, 100, 0])


def test_segment_box_round_trip():
    for seg in ([10, 50, 200, 50], [30, 10, 30, 300]):
        assert g.box_to_segment(g.segment_to_box(seg)) == pytest.approx(seg)
    assert g.box_to_segment(g.segment_to_box([0, 100, 100, 0]), diagonal="up") == pytest.approx([0, 100, 100, 0])
