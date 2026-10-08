import json

import httpx

from drawing_ai.merge import PatchNetwork, merge_text
from drawing_ai.merge.llm import LLMMergeClient, parse_decision
from drawing_ai.schemas import AnnotationArrow, ObjectComponent, OCRItem, TileAnnotation

# settings fixture: canvas 960, tiles 512, overlap 64 -> tile x/y starts [0, 448]
LEFT = [0, 0, 512, 512]
RIGHT = [448, 0, 512, 512]


def _tiles(left_items, right_items):
    a = TileAnnotation(image_id="IMG_001", tile_id="Tile_001", tile_box=LEFT, **left_items)
    b = TileAnnotation(image_id="IMG_001", tile_id="Tile_002", tile_box=RIGHT, **right_items)
    return [a, b]


def test_line_split_across_seam_is_merged(settings):
    # canvas line 300 -> 700 at y=100; each tile sees its clipped piece
    tiles = _tiles(
        {"arrows": [AnnotationArrow(points=[300, 100, 511, 100], class_id="leader_line", confidence=0.9)]},
        {"arrows": [AnnotationArrow(points=[0, 100, 252, 100], class_id="dimension_line", confidence=0.6)]},
    )
    merged, report = PatchNetwork(settings).merge(tiles)
    assert len(merged.arrows) == 1
    a = merged.arrows[0]
    assert a.points == [300, 100, 700, 100]
    assert a.class_id == "leader_line"  # confidence-weighted vote
    assert a.source_tiles == ["Tile_001", "Tile_002"]
    assert report["merged_groups"] == 1


def test_parallel_lines_are_not_merged(settings):
    tiles = _tiles(
        {"arrows": [AnnotationArrow(points=[300, 100, 511, 100], class_id="dimension_line")]},
        {"arrows": [AnnotationArrow(points=[0, 140, 252, 140], class_id="dimension_line")]},
    )
    merged, _ = PatchNetwork(settings).merge(tiles)
    assert len(merged.arrows) == 2


def test_duplicate_box_in_overlap_and_cut_box(settings):
    # object fully in overlap seen twice + object cut by the seam
    tiles = _tiles(
        {"objects": [ObjectComponent(bbox=[460, 300, 40, 40], classification_type="Ctype_1", confidence=0.8),
                     ObjectComponent(bbox=[400, 20, 112, 60], classification_type="Ctype_2", confidence=0.7)]},
        {"objects": [ObjectComponent(bbox=[12, 300, 40, 40], classification_type="Ctype_1", confidence=0.9),
                     ObjectComponent(bbox=[0, 20, 100, 60], classification_type="Ctype_2", confidence=0.8)]},
    )
    merged, _ = PatchNetwork(settings).merge(tiles)
    boxes = sorted(o.bbox for o in merged.objects)
    assert boxes == [[400, 20, 148, 60], [460, 300, 40, 40]]


def test_ocr_fragments_joined(settings):
    tiles = _tiles(
        {"ocr": [OCRItem(bbox=[470, 200, 42, 20], text="Ø25.", confidence=0.8)]},
        {"ocr": [OCRItem(bbox=[30, 200, 40, 20], text="25.4", confidence=0.8)]},
    )
    merged, _ = PatchNetwork(settings).merge(tiles)
    assert [o.text for o in merged.ocr] == ["Ø25.4"]


def test_merge_text():
    assert merge_text("Ø25.", "25.4") == "Ø25.4"
    assert merge_text("THRU", "THRU") == "THRU"
    assert merge_text("4X", "M6") == "4X M6"


def test_parse_decision_drops_invalid_ids():
    from drawing_ai.merge.candidates import Candidate

    known = {
        "T1:arrow:0": Candidate("T1:arrow:0", "arrow", "T1", [0, 0, 1, 1], "a"),
        "T2:arrow:0": Candidate("T2:arrow:0", "arrow", "T2", [0, 0, 1, 1], "b"),
        "T2:ocr:0": Candidate("T2:ocr:0", "ocr", "T2", [0, 0, 1, 1], "x"),
    }
    d = parse_decision(json.dumps({"merges": [{"ids": ["T1:arrow:0", "T2:arrow:0"], "label": "leader_line"},
                                              {"ids": ["T1:arrow:0", "T2:ocr:0"]},
                                              {"ids": ["T1:arrow:0", "nope"]}],
                                   "relabels": [{"id": "T2:ocr:0", "label": "07"}]}), known)
    assert [m.ids for m in d.merges] == [["T1:arrow:0", "T2:arrow:0"]]
    assert d.relabels == {"T2:ocr:0": "07"}


def test_llm_backend_with_mock_endpoint(settings):
    seen = {}

    def handler(request: httpx.Request):
        body = json.loads(request.content)
        payload = json.loads(body["messages"][1]["content"])
        seen["payload"] = payload
        ids = [payload["tile_a"]["items"][0]["id"], payload["tile_b"]["items"][0]["id"]]
        content = json.dumps({"merges": [{"ids": ids, "label": "section_arrow"}], "relabels": []})
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    s = settings.model_copy(update={"llm_base_url": "http://llm.test/v1", "merge_backend": "llm"})
    client = LLMMergeClient(s, client=httpx.Client(transport=httpx.MockTransport(handler)))
    tiles = _tiles(
        {"arrows": [AnnotationArrow(points=[300, 100, 511, 100], class_id="leader_line", confidence=0.9)]},
        {"arrows": [AnnotationArrow(points=[0, 100, 252, 100], class_id="dimension_line", confidence=0.6)]},
    )
    merged, report = PatchNetwork(s, llm=client).merge(tiles)
    assert report["backend"] == "llm" and report["llm_calls"] == 1 and report["llm_failures"] == 0
    assert seen["payload"]["heuristic_suggestions"]  # heuristic hints are passed to the LLM
    assert len(merged.arrows) == 1 and merged.arrows[0].class_id == "section_arrow"


def test_llm_failure_falls_back_to_heuristic(settings):
    s = settings.model_copy(update={"llm_base_url": "http://llm.test/v1"})
    client = LLMMergeClient(s, client=httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(500))))
    tiles = _tiles(
        {"arrows": [AnnotationArrow(points=[300, 100, 511, 100], class_id="leader_line")]},
        {"arrows": [AnnotationArrow(points=[0, 100, 252, 100], class_id="leader_line")]},
    )
    merged, report = PatchNetwork(s, llm=client).merge(tiles)
    assert report["llm_failures"] == 1
    assert len(merged.arrows) == 1
