"""
### housing_pipeline

Una ejecución = una petición a la API de datos = un lote.

1. Pide un lote, lo guarda tal cual en MinIO (`landing`) y en `raw.properties`.
2. Según el resultado: lote nuevo → validación y decisión; lote repetido → monitoreo del
   modelo productivo; fin de datos → reinicio de la API (política `restart`).
3. Valida esquema, calidad, categorías nuevas y drift; guarda las filas válidas en CLEAN.
4. Decide si entrenar con reglas técnicas (`housing_ml.decision`).
5. Entrena, evalúa y registra el candidato en MLflow, lo compara con el modelo productivo
   sobre el mismo holdout (`housing_ml.gates`) y lo promueve o rechaza.
6. Deja el resultado en `audit.batch_runs`, que muestra Streamlit.
"""

import os
from datetime import datetime, timedelta

from airflow.providers.standard.operators.empty import EmptyOperator
from airflow.providers.standard.operators.python import BranchPythonOperator, PythonOperator
from airflow.sdk import DAG

from pipeline import tasks

default_args = {
    "owner": "housing-mlops",
    "retries": 2,
    "retry_delay": timedelta(minutes=1),
    "retry_exponential_backoff": True,
    "execution_timeout": timedelta(minutes=30),
    "on_failure_callback": tasks.on_task_failure,
}

with DAG(
    dag_id="housing_pipeline",
    description="Ingesta por lotes, validación, decisión de entrenamiento y promoción del modelo",
    doc_md=__doc__,
    start_date=datetime(2026, 11, 1),
    schedule=os.environ.get("PIPELINE_SCHEDULE", "*/15 * * * *"),
    catchup=False,
    max_active_runs=1,
    is_paused_upon_creation=True,
    default_args=default_args,
    tags=["housing", "mlops"],
) as dag:

    def python(task_id, callable_, **kwargs):
        return PythonOperator(task_id=task_id, python_callable=callable_, **kwargs)

    def branch(task_id, callable_):
        return BranchPythonOperator(task_id=task_id, python_callable=callable_)

    start = EmptyOperator(task_id="start")
    check_dependencies = python("check_dependencies", tasks.check_dependencies, retries=3)
    fetch = python("fetch_batch_from_api", tasks.fetch_batch_from_api, retries=3)
    store = python("store_raw_batch", tasks.store_raw_batch)
    route = branch("route_batch", tasks.route_batch)

    validate_schema = python("validate_schema", tasks.validate_schema)
    validate_quality = python("validate_data_quality", tasks.validate_data_quality)
    new_categories = python("detect_new_categories", tasks.detect_new_categories)
    drift = python("detect_data_drift", tasks.detect_data_drift)
    preprocess = python("preprocess_data", tasks.preprocess_data)
    evaluate_production = python("evaluate_production_on_batch", tasks.evaluate_production_on_batch)
    export_replay = python("export_replay_sample", tasks.export_replay_sample)

    decide_training = branch("decide_training", tasks.decide_training_branch)
    skip_training = python("skip_training", tasks.skip_training)
    train = python(
        "train_candidate_model",
        tasks.train_candidate_model,
        retries=1,
        execution_timeout=timedelta(minutes=60),
    )
    evaluate_candidate = python("evaluate_candidate_model", tasks.evaluate_candidate_model)
    register = python("register_candidate_in_mlflow", tasks.register_candidate_in_mlflow)
    compare = python("compare_with_production", tasks.compare_with_production)
    decide_promotion = branch("decide_promotion", tasks.decide_promotion_branch)
    promote = python("promote_model", tasks.promote_model)
    reject = python("reject_model", tasks.reject_model)

    monitor = python("monitor_production_model", tasks.monitor_production_model)
    end_of_data = python("handle_end_of_data", tasks.handle_end_of_data)
    empty = python("record_empty_batch", tasks.record_empty_batch)

    notify = python(
        "notify_or_log_result",
        tasks.notify_or_log_result,
        trigger_rule="none_failed_min_one_success",
    )
    end = EmptyOperator(task_id="end")

    start >> check_dependencies >> fetch >> store >> route
    route >> [validate_schema, monitor, end_of_data, empty]
    validate_schema >> validate_quality >> new_categories >> drift >> preprocess
    preprocess >> [evaluate_production, export_replay]
    evaluate_production >> decide_training >> [skip_training, train]
    train >> evaluate_candidate >> register >> compare >> decide_promotion >> [promote, reject]
    [skip_training, promote, reject, export_replay, monitor, end_of_data, empty] >> notify >> end
