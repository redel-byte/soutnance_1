import subprocess
import pendulum

from airflow.sdk import dag, task, chain, get_current_context


@dag(
    dag_id="student_risk_pipeline",
    schedule="@daily",
    start_date=pendulum.datetime(2026, 1, 1, tz="Africa/Casablanca"),
    catchup=False,
    max_active_runs=1,
)
def student_risk_pipeline():

    @task
    def run_stage(stage):
        run_id = get_current_context()["run_id"]

        command = [
            "/opt/airflow/ml-venv/bin/python",
            "-m", "soutnance_1.pipeline",
            stage,
            run_id,
        ]

        subprocess.run(command, check=True)

    stages = ["prepare", "preprocess", "train", "evaluate"]

    tasks = [
        run_stage.override(task_id=stage)(stage)
        for stage in stages
    ]

    chain(*tasks)


student_risk_pipeline()
