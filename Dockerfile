FROM apache/airflow:slim-3.4.0-python3.14

WORKDIR /opt/project

COPY requirements-ml.txt app/requirements.txt /tmp/
RUN pip install --no-cache-dir "apache-airflow==${AIRFLOW_VERSION}" \
    -r /tmp/requirements-ml.txt -r /tmp/requirements.txt \
    && pip check \
    && mkdir -p /home/airflow/student-artifacts

ENV PYTHONPATH=/opt/project/src \
    AIRFLOW__CORE__DAGS_FOLDER=/opt/project/dags \
    STUDENT_OUTPUT_DIR=/home/airflow/student-artifacts \
    OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=1

COPY src ./src
COPY dags ./dags
COPY app ./app
COPY .streamlit ./.streamlit
