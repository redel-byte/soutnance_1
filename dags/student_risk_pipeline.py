import logging
import subprocess
from datetime import timedelta

import pendulum
from airflow.sdk import dag, get_current_context, task


def run_stage(stage: str, *options: str) -> None:
    command = [
        "/opt/airflow/ml-venv/bin/python", "-u", "-m", "soutnance_1.pipeline",
        stage, get_current_context()["run_id"], *options,
    ]
    logger = logging.getLogger(__name__)
    with subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    ) as process:
        for line in process.stdout:
            logger.info("%s", line.rstrip())
        if process.wait():
            raise subprocess.CalledProcessError(process.returncode, command)


@dag(
    dag_id="student_risk_pipeline",
    description="Prépare les données, entraîne et évalue le modèle de risque étudiant.",
    schedule="0 6 * * *",
    start_date=pendulum.datetime(2026, 1, 1, tz="Africa/Casablanca"),
    catchup=False,
    max_active_runs=1,
    default_args={"retries": 1, "retry_delay": timedelta(minutes=2)},
    tags=["classification", "student-risk"],
)
def student_risk_pipeline():
    @task
    def prepare_data():
        run_stage("prepare")

    @task
    def preprocess_data():
        run_stage("preprocess")

    @task(execution_timeout=timedelta(minutes=30))
    def train_model():
        run_stage("train")

    @task
    def evaluate_model():
        dag_run = get_current_context()["dag_run"]
        options = ["--test-final"] if (dag_run.conf or {}).get("test_final") is True else []
        run_stage("evaluate", *options)

    prepare_data() >> preprocess_data() >> train_model() >> evaluate_model()


dag_instance = student_risk_pipeline()
