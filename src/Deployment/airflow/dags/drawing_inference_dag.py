"""Airflow DAG: Preprocessing -> Inference -> Post-processing pipelines.

Triggered per job by the Celery orchestrator (PIPELINE_EXECUTOR=airflow) with
conf {job_id, input_file_id, filename, image_id}. Each task calls the same
stage function the inline executor uses; state is passed through MongoDB.
"""

import pendulum
from airflow.decorators import dag, task


@dag(
    dag_id="drawing_inference",
    schedule=None,
    start_date=pendulum.datetime(2024, 1, 1, tz="UTC"),
    catchup=False,
    max_active_runs=8,
    default_args={"retries": 1},
    tags=["drawing-ai", "inference"],
)
def drawing_inference():
    @task
    def preprocessing_pipeline(**ctx):
        from drawing_ai.pipeline import stage_preprocess

        c = ctx["dag_run"].conf
        stage_preprocess(c["job_id"], c["input_file_id"], c["filename"], c.get("image_id"))
        return c["job_id"]

    @task
    def inference_pipeline(job_id: str):
        from drawing_ai.pipeline import stage_infer

        stage_infer(job_id)
        return job_id

    @task
    def post_processing(job_id: str):
        from drawing_ai.pipeline import stage_postprocess

        return stage_postprocess(job_id)

    post_processing(inference_pipeline(preprocessing_pipeline()))


drawing_inference()
