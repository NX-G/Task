# Airflow image with the drawing_ai pipeline installed.
# Build context: repository root.
FROM apache/airflow:2.10.5-python3.11

USER root
RUN apt-get update && apt-get install -y --no-install-recommends tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*
USER airflow

COPY src/Deployment/requirements.txt /tmp/requirements.txt
# Keep numpy 1.x: Airflow's bundled pandas is compiled against it.
RUN pip install --no-cache-dir -r /tmp/requirements.txt "numpy<2" "opencv-python-headless<4.11"

COPY --chown=airflow:root src/drawing_ai /opt/drawing/src/drawing_ai
COPY --chown=airflow:root src/Deployment/airflow/dags /opt/airflow/dags
ENV PYTHONPATH=/opt/drawing/src
