import subprocess
import sys

import pendulum

from airflow.sdk import chain, dag, task


@dag(
    dag_id="student_risk_pipeline",
    schedule="@daily",
    start_date=pendulum.datetime(2026, 1, 1, tz="Africa/Casablanca"),
    catchup=False,
    max_active_runs=1,
)
def student_risk_pipeline():
    @task
    def run_stage(module):
        subprocess.run([sys.executable, "-u", "-m", f"soutnance_1.{module}"], check=True)

    stages = {
        "prepare": "preparation",
        "preprocess": "pretraitement",
        "train": "entrainement",
        "evaluate": "evaluation",
    }
    tasks = [
        run_stage.override(task_id=stage)(module)
        for stage, module in stages.items()
    ]

    chain(*tasks)


student_risk_pipeline()
