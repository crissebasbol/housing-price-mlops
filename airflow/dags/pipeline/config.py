import os

DATA_API_URL = os.environ.get("DATA_API_URL", "http://data-api")
DATA_API_GROUP = int(os.environ.get("DATA_API_GROUP", "1"))
DATA_API_DAY = os.environ.get("DATA_API_DAY", "Tuesday")
DATA_API_TIMEOUT = float(os.environ.get("DATA_API_TIMEOUT", "180"))
END_OF_DATA_POLICY = os.environ.get("END_OF_DATA_POLICY", "restart")
MAX_RECOVERY_ATTEMPTS = int(os.environ.get("MAX_RECOVERY_ATTEMPTS", "3"))

DATA_DB = {
    "host": os.environ.get("DATA_DB_HOST", "postgres-data"),
    "port": int(os.environ.get("DATA_DB_PORT", "5432")),
    "dbname": os.environ.get("DATA_DB_NAME", "housing"),
    "user": os.environ.get("DATA_DB_USER", "airflow_etl"),
}

S3_ENDPOINT_URL = os.environ.get("S3_ENDPOINT_URL", "http://minio:9000")
LANDING_BUCKET = os.environ.get("LANDING_BUCKET", "landing")
DATASETS_BUCKET = os.environ.get("DATASETS_BUCKET", "datasets")
REPLAY_OBJECT = "replay/holdout_sample.json"
REPLAY_SAMPLE_ROWS = int(os.environ.get("REPLAY_SAMPLE_ROWS", "5000"))

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MLFLOW_EXPERIMENT = os.environ.get("MLFLOW_EXPERIMENT", "housing-price")
MODEL_NAME = os.environ.get("MODEL_NAME", "housing-price-model")
CHAMPION = "champion"
CHALLENGER = "challenger"
PREVIOUS_CHAMPION = "previous_champion"
SHAP_ROWS = int(os.environ.get("SHAP_ROWS", "2000"))

GIT_SHA = os.environ.get("GIT_SHA", "unknown")
INFERENCE_SOURCES_FOR_DRIFT = ("streamlit", "api")


def data_db_password() -> str:
    password = os.environ.get("DATA_DB_PASSWORD")
    if not password:
        raise RuntimeError("Falta DATA_DB_PASSWORD (Secret postgres-data)")
    return password
