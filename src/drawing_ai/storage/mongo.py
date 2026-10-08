"""MongoDB artifact store.

- GridFS bucket ``files``: uploaded drawings, normalised canvases, visualisations.
- ``pipeline_state``: per-job intermediate state handed between pipeline stages
  (so preprocessing / inference / post-processing can run as separate Airflow
  tasks on different workers).
- ``results``: final JSON per job (inference + eval data).
- ``drift``: per-page drift profiles and comparisons.
- ``baselines``: drift baselines written by training post-processing.
- ``training_images``: optional training data source (GridFS id + annotation).
"""

import datetime as dt
from functools import lru_cache
from typing import Any, Dict, List, Optional

import gridfs
from bson import ObjectId
from pymongo import DESCENDING, MongoClient


def _now():
    return dt.datetime.now(dt.timezone.utc)


class ArtifactStore:
    def __init__(self, uri: str, db: str):
        self.client = MongoClient(uri, tz_aware=True, serverSelectionTimeoutMS=5000)
        self.db = self.client[db]
        self.fs = gridfs.GridFS(self.db, collection="files")
        self.db.pipeline_state.create_index("job_id", unique=True)
        self.db.results.create_index("job_id", unique=True)
        self.db.drift.create_index([("created_at", DESCENDING)])

    # --- files
    def put_file(self, data: bytes, filename: str, content_type: str = "application/octet-stream", **meta) -> str:
        return str(self.fs.put(data, filename=filename, content_type=content_type, metadata=meta))

    def get_file(self, file_id: str) -> bytes:
        return self.fs.get(ObjectId(file_id)).read()

    def file_info(self, file_id: str) -> Dict[str, Any]:
        f = self.fs.get(ObjectId(file_id))
        return {"filename": f.filename, "content_type": getattr(f, "content_type", None), "length": f.length}

    # --- pipeline state
    def init_state(self, job_id: str, doc: dict) -> None:
        self.db.pipeline_state.replace_one({"job_id": job_id}, {"job_id": job_id, "created_at": _now(), **doc}, upsert=True)

    def update_state(self, job_id: str, fields: dict) -> None:
        self.db.pipeline_state.update_one({"job_id": job_id}, {"$set": {**fields, "updated_at": _now()}})

    def get_state(self, job_id: str) -> Optional[dict]:
        return self.db.pipeline_state.find_one({"job_id": job_id}, {"_id": 0})

    # --- results
    def save_result(self, job_id: str, doc: dict) -> None:
        self.db.results.replace_one({"job_id": job_id}, {"job_id": job_id, "created_at": _now(), **doc}, upsert=True)

    def get_result(self, job_id: str) -> Optional[dict]:
        return self.db.results.find_one({"job_id": job_id}, {"_id": 0})

    # --- drift
    def save_drift(self, doc: dict) -> None:
        self.db.drift.insert_one({"created_at": _now(), **doc})

    def recent_profiles(self, limit: int) -> List[dict]:
        cur = self.db.drift.find({}, {"profile": 1}).sort("created_at", DESCENDING).limit(limit)
        return [d["profile"] for d in cur if d.get("profile")]

    def save_baseline(self, name: str, profile: dict, meta: dict) -> None:
        self.db.baselines.insert_one({"name": name, "profile": profile, "meta": meta, "created_at": _now()})

    def latest_baseline(self, name: str = "default") -> Optional[dict]:
        doc = self.db.baselines.find_one({"name": name}, sort=[("created_at", DESCENDING)])
        return doc["profile"] if doc else None

    def ping(self) -> bool:
        self.client.admin.command("ping")
        return True


@lru_cache
def get_store(uri: str, db: str) -> ArtifactStore:
    return ArtifactStore(uri, db)
