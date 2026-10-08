"""Resolve trained weights from the MLflow Model Registry.

Training logs each model's files under the run artifact path ``model/`` and
registers ``runs:/<run_id>/model`` under a registered-model name. Promotion sets
the ``champion`` alias. Inference resolves ``<name>@<alias>`` and downloads the
run artifacts to a local cache keyed by model version.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)


@dataclass
class ResolvedModel:
    name: str
    version: str
    run_id: str
    local_dir: Path


def resolve_model(tracking_uri: str, name: str, alias: str, cache_dir: str) -> Optional[ResolvedModel]:
    try:
        import mlflow
        from mlflow import MlflowClient
    except ImportError:
        log.info("mlflow not installed; cannot resolve %s", name)
        return None
    try:
        mlflow.set_tracking_uri(tracking_uri)
        mv = MlflowClient().get_model_version_by_alias(name, alias)
    except Exception as exc:  # registry unreachable or model not registered yet
        log.info("No %s@%s in MLflow (%s)", name, alias, type(exc).__name__)
        return None
    target = Path(cache_dir) / name / str(mv.version)
    if not (target / ".complete").exists():
        target.mkdir(parents=True, exist_ok=True)
        mlflow.artifacts.download_artifacts(run_id=mv.run_id, artifact_path="model", dst_path=str(target))
        (target / ".complete").touch()
    return ResolvedModel(name, str(mv.version), mv.run_id, target / "model")
