import re
from typing import Dict

import mlflow

EXPERIMENT = "drawing-ai"


def setup(tracking_uri: str, experiment: str = EXPERIMENT) -> None:
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)


def clean_metrics(d: Dict) -> Dict[str, float]:
    """MLflow metric keys only allow [A-Za-z0-9_\\-. /]."""
    out = {}
    for k, v in d.items():
        try:
            out[re.sub(r"[^A-Za-z0-9_\-. /]", "_", str(k)).strip("_")] = float(v)
        except (TypeError, ValueError):
            continue
    return out
