FROM apache/airflow:3.2.2-python3.12 AS airflow

USER root
RUN install -d -o airflow -g root -m 0775 \
    /opt/airflow/work /opt/airflow/results /opt/airflow/auth /opt/airflow/ml-venv
USER airflow

ENV PYTHONPATH=/opt/project/src \
    OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=1

COPY requirements-ml.txt /tmp/requirements-ml.txt
RUN python -m venv /opt/airflow/ml-venv \
    && /opt/airflow/ml-venv/bin/pip install --no-cache-dir -r /tmp/requirements-ml.txt \
    && /opt/airflow/ml-venv/bin/python -c "from sklearn.pipeline import Pipeline; import joblib, pandas, scipy"

COPY --chown=airflow:root src /opt/project/src
COPY --chown=airflow:root dags /opt/airflow/dags
RUN test -s /opt/airflow/dags/student_risk_pipeline.py \
    && /opt/airflow/ml-venv/bin/python -c "from soutnance_1.pipeline import main"

# Use updated IANA timezone rules for Airflow's scheduler and DAG processor.
RUN pip install --no-cache-dir tzdata==2026.6
ENV PYTHONTZPATH=""

FROM python:3.12-slim-bookworm AS streamlit

WORKDIR /opt/project
COPY requirements-ml.txt ./
COPY app/requirements.txt ./requirements-streamlit.txt
RUN pip install --no-cache-dir -r requirements-ml.txt -r requirements-streamlit.txt \
    && useradd --create-home --uid 10001 streamlit

COPY src ./src
COPY app ./app
COPY .streamlit ./.streamlit
ENV PYTHONPATH=/opt/project/src \
    OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=1
USER streamlit

CMD ["streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
