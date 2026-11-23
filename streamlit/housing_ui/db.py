import pandas as pd
import psycopg2

from housing_ui.config import SETTINGS

BATCH_RUNS = """
SELECT ingestion_id, dag_run_id, started_at, finished_at, cycle, api_batch_number, outcome,
       record_count, valid_rows, rejected_rows, schema_status, schema_report, quality_report,
       new_categories, drift_report, champion_version_before, champion_mape_on_batch,
       training_decision, training_code, training_signals, training_reason, mlflow_run_id,
       candidate_version, gates, diagnostics, promotion_decision, promotion_reason,
       mape_candidate, mape_champion, mape_delta_pct, champion_version_after, final_status, error
FROM audit.batch_runs
ORDER BY ingestion_id
"""

MODEL_ACTIONS = """
SELECT ts, action, from_version, to_version, actor, reason, source
FROM audit.model_actions
ORDER BY ts DESC
LIMIT 30
"""


def query(sql: str) -> pd.DataFrame:
    conn = psycopg2.connect(
        host=SETTINGS.db_host,
        port=SETTINGS.db_port,
        dbname=SETTINGS.db_name,
        user=SETTINGS.db_user,
        password=SETTINGS.db_password,
        connect_timeout=5,
    )
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
            columns = [col.name for col in cur.description]
            return pd.DataFrame(cur.fetchall(), columns=columns)
    finally:
        conn.close()


def batch_runs() -> pd.DataFrame:
    return query(BATCH_RUNS)


def model_actions() -> pd.DataFrame:
    return query(MODEL_ACTIONS)
