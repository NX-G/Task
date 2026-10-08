"""Pull training data from the MongoDB image store into the raw directory
layout expected by feature engineering.

Collection ``training_images`` documents:
    {"image_id": "IMG_001", "file_id": <GridFS id>, "filename": "IMG_001.png",
     "annotation": {...spec JSON, image-level or list of tiles...}}
"""

import json
from pathlib import Path

from drawing_ai.config import Settings
from drawing_ai.storage.mongo import get_store


def export_from_mongo(raw_dir: Path, s: Settings) -> int:
    store = get_store(s.mongo_uri, s.mongo_db)
    (raw_dir / "images").mkdir(parents=True, exist_ok=True)
    (raw_dir / "annotations").mkdir(parents=True, exist_ok=True)
    n = 0
    for doc in store.db.training_images.find({"annotation": {"$exists": True}}):
        ext = Path(doc.get("filename", ".png")).suffix or ".png"
        (raw_dir / "images" / f"{doc['image_id']}{ext}").write_bytes(store.get_file(str(doc["file_id"])))
        (raw_dir / "annotations" / f"{doc['image_id']}.json").write_text(json.dumps(doc["annotation"], ensure_ascii=False))
        n += 1
    return n
