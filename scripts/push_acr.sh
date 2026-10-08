#!/usr/bin/env bash
# Tag and push all images to Azure Container Registry.
#   ACR_NAME=myregistry TAG=v0.1.0 ./scripts/push_acr.sh
set -euo pipefail
: "${ACR_NAME:?set ACR_NAME}"
TAG="${TAG:-$(date +%Y%m%d%H%M)}"
REGISTRY="${ACR_NAME}.azurecr.io"
az acr login --name "$ACR_NAME"
for img in service airflow mlflow trainer; do
  if docker image inspect "drawing-ai/${img}:latest" >/dev/null 2>&1; then
    docker tag "drawing-ai/${img}:latest" "${REGISTRY}/drawing-ai/${img}:${TAG}"
    docker push "${REGISTRY}/drawing-ai/${img}:${TAG}"
  else
    echo "skip ${img} (not built)"
  fi
done
