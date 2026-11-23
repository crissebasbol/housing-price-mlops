FROM apache/airflow:3.1.8

ARG GIT_SHA=unknown
ENV GIT_SHA=${GIT_SHA} \
    MLFLOW_ENABLE_PROXY_MULTIPART_DOWNLOAD=false \
    PYTHONPATH=/opt/airflow/dags

COPY --chown=airflow:root airflow/requirements.txt /tmp/requirements.txt
COPY --chown=airflow:root housing_ml /tmp/housing_ml
RUN pip install --no-cache-dir "apache-airflow==${AIRFLOW_VERSION}" \
    -r /tmp/requirements.txt /tmp/housing_ml

COPY --chown=airflow:root airflow/dags /opt/airflow/dags

RUN python /opt/airflow/dags/housing_pipeline.py
