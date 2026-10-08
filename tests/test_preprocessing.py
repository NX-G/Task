import numpy as np
import pytest

from drawing_ai.preprocessing import preprocess, tile_grid, tile_starts


def test_tile_starts_cover_edge():
    assert tile_starts(4096, 1024, 128) == [0, 896, 1792, 2688, 3072]
    assert tile_starts(512, 1024, 128) == [0]


def test_default_grid_is_row_major_and_complete():
    grid = tile_grid(4096, 2896, 1024, 128)
    assert grid[0]["tile_id"] == "Tile_001" and grid[0]["box"] == [0, 0, 1024, 1024]
    assert len(grid) == 5 * 4
    covered = np.zeros((2896, 4096), bool)
    for t in grid:
        x, y, w, h = t["box"]
        covered[y : y + h, x : x + w] = True
    assert covered.all()


def test_fixed_canvas_and_transform_round_trip(settings):
    img = np.full((600, 300, 3), 255, np.uint8)  # portrait -> rotated to landscape
    pre = preprocess(img, settings)
    tf = pre.transform
    assert pre.canvas.shape == (settings.canvas_height, settings.canvas_width, 3)
    assert tf.rotated
    for pt in [(0, 0), (299, 599), (120, 40)]:
        assert tf.to_original(*tf.to_canvas(*pt)) == pytest.approx(pt, abs=1e-6)
    assert len(pre.tiles) == 4
    assert all(t.image.shape[:2] == (t.h, t.w) for t in pre.tiles)
