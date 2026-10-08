import pytest

from drawing_ai.config import Settings


@pytest.fixture
def settings():
    # small geometry keeps tests fast; 2x2 tiles with a 64px overlap
    return Settings(canvas_width=960, canvas_height=960, tile_size=512, tile_overlap=64, llm_base_url="", merge_backend="heuristic")
