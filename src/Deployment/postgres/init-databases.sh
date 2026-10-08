#!/bin/bash
# Creates the extra databases used by MLflow and Airflow on first start.
set -euo pipefail
for db in mlflow airflow; do
  psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" \
    -c "CREATE DATABASE $db OWNER $POSTGRES_USER;"
done
