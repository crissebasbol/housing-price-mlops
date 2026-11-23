import csv
import io
import json
import math
from contextlib import contextmanager
from datetime import date, datetime

import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import Json, RealDictCursor

from pipeline.config import DATA_DB, data_db_password

REQUIRED_TABLES = (
    "raw.batches",
    "raw.properties",
    "raw.inference_events",
    "clean.properties",
    "audit.batch_runs",
    "audit.model_evaluations",
)
CLEAN_COLUMNS = (
    "row_hash",
    "ingestion_id",
    "cycle",
    "api_batch_number",
    "split",
    "price",
    "bed",
    "bath",
    "acre_lot",
    "house_size",
    "prev_sold_date",
    "status",
    "city",
    "state",
    "zip_code",
    "brokered_by",
    "preprocessing_version",
)
TEXT_COLUMNS = ("row_hash", "split", "status", "city", "state", "zip_code", "brokered_by")
CATEGORY_COLUMNS = ("status", "state", "city", "zip_code", "brokered_by")
RECOVERABLE_STATUSES = ("fetched", "stored", "failed")


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (datetime, date, pd.Timestamp)):
        return value.isoformat()
    return str(value)


def as_json(value) -> Json:
    return Json(value, dumps=lambda v: json.dumps(v, default=_json_default, ensure_ascii=False))


def _cell(value):
    if value is None or value is pd.NaT:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, np.generic):
        return _cell(value.item())
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if isinstance(value, (list, tuple)):
        return "{" + ",".join(str(v) for v in value) + "}"
    if isinstance(value, dict):
        return json.dumps(value, default=_json_default, ensure_ascii=False)
    return value


def _csv_buffer(rows) -> io.StringIO:
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    for row in rows:
        writer.writerow(["" if (cell := _cell(v)) is None else cell for v in row])
    buffer.seek(0)
    return buffer


class Database:
    """SQL contra la base `housing` (schemas raw, clean y audit) con el rol airflow_etl."""

    @contextmanager
    def cursor(self, dict_rows: bool = False):
        conn = psycopg2.connect(**DATA_DB, password=data_db_password(), connect_timeout=10)
        try:
            factory = RealDictCursor if dict_rows else None
            with conn, conn.cursor(cursor_factory=factory) as cur:
                yield cur
        finally:
            conn.close()

    def check(self) -> None:
        with self.cursor() as cur:
            missing = []
            for table in REQUIRED_TABLES:
                cur.execute("SELECT to_regclass(%s)", (table,))
                if cur.fetchone()[0] is None:
                    missing.append(table)
        if missing:
            raise RuntimeError(f"Faltan tablas en la base housing: {missing}")

    def one(self, sql: str, params=None) -> dict | None:
        with self.cursor(dict_rows=True) as cur:
            cur.execute(sql, params)
            row = cur.fetchone()
            return dict(row) if row else None

    def all(self, sql: str, params=None) -> list[dict]:
        with self.cursor(dict_rows=True) as cur:
            cur.execute(sql, params)
            return [dict(row) for row in cur.fetchall()]

    def scalar(self, sql: str, params=None):
        with self.cursor() as cur:
            cur.execute(sql, params)
            row = cur.fetchone()
            return row[0] if row else None

    def read_frame(self, sql: str, params=None) -> pd.DataFrame:
        buffer = io.StringIO()
        with self.cursor() as cur:
            query = cur.mogrify(sql, params).decode()
            cur.copy_expert(f"COPY ({query}) TO STDOUT WITH CSV HEADER", buffer)
        buffer.seek(0)
        frame = pd.read_csv(buffer, dtype={c: str for c in TEXT_COLUMNS})
        if "prev_sold_date" in frame.columns:
            frame["prev_sold_date"] = pd.to_datetime(frame["prev_sold_date"], errors="coerce")
        return frame

    def _set_clause(self, fields: dict) -> tuple[str, list]:
        values = [as_json(v) if isinstance(v, dict) else v for v in fields.values()]
        return ", ".join(f"{name} = %s" for name in fields), values

    # ------------------------------------------------------------- raw.batches
    def insert_batch(self, **fields) -> int:
        columns = ", ".join(fields)
        placeholders = ", ".join(["%s"] * len(fields))
        values = [as_json(v) if isinstance(v, dict) else v for v in fields.values()]
        with self.cursor() as cur:
            cur.execute(
                f"INSERT INTO raw.batches ({columns}) VALUES ({placeholders}) "
                "RETURNING ingestion_id",
                values,
            )
            return cur.fetchone()[0]

    def update_batch(self, ingestion_id: int, **fields) -> None:
        clause, values = self._set_clause(fields)
        with self.cursor() as cur:
            cur.execute(
                f"UPDATE raw.batches SET {clause}, updated_at = now() WHERE ingestion_id = %s",
                [*values, ingestion_id],
            )

    def batch(self, ingestion_id: int) -> dict | None:
        return self.one("SELECT * FROM raw.batches WHERE ingestion_id = %s", (ingestion_id,))

    def batch_for_run(self, dag_run_id: str) -> dict | None:
        return self.one(
            "SELECT * FROM raw.batches WHERE dag_run_id = %s ORDER BY ingestion_id DESC LIMIT 1",
            (dag_run_id,),
        )

    def recoverable_batch(self, max_attempts: int) -> dict | None:
        return self.one(
            "SELECT * FROM raw.batches WHERE outcome = 'new' AND status IN %s AND attempts < %s "
            "ORDER BY ingestion_id LIMIT 1",
            (RECOVERABLE_STATUSES, max_attempts),
        )

    def quarantine_exhausted(self, max_attempts: int) -> int:
        with self.cursor() as cur:
            cur.execute(
                "UPDATE raw.batches SET status = 'quarantined', updated_at = now() "
                "WHERE outcome = 'new' AND status IN %s AND attempts >= %s",
                (RECOVERABLE_STATUSES, max_attempts),
            )
            return cur.rowcount

    def last_batch(self) -> dict | None:
        return self.one("SELECT * FROM raw.batches ORDER BY ingestion_id DESC LIMIT 1")

    def next_expected_batch(self, cycle: int) -> int:
        last = self.scalar(
            "SELECT max(api_batch_number) FROM raw.batches "
            "WHERE cycle = %s AND outcome IN ('new', 'duplicate', 'empty')",
            (cycle,),
        )
        return 0 if last is None else int(last) + 1

    def find_new_batch(self, data_sha256: str) -> int | None:
        return self.scalar(
            "SELECT ingestion_id FROM raw.batches WHERE outcome = 'new' AND data_sha256 = %s",
            (data_sha256,),
        )

    # ------------------------------------------------------------- raw.properties
    def replace_raw_properties(self, ingestion_id: int, records: list[dict], hashes: list[str]):
        rows = (
            (ingestion_id, position, row_hash, json.dumps(record, ensure_ascii=False))
            for position, (record, row_hash) in enumerate(zip(records, hashes, strict=True))
        )
        with self.cursor() as cur:
            cur.execute("DELETE FROM raw.properties WHERE ingestion_id = %s", (ingestion_id,))
            cur.copy_expert(
                "COPY raw.properties (ingestion_id, source_row, row_hash, payload) "
                "FROM STDIN WITH CSV",
                _csv_buffer(rows),
            )

    def set_quality(self, ingestion_id: int, statuses: list[tuple[int, str, list[str]]]) -> None:
        with self.cursor() as cur:
            cur.execute(
                "CREATE TEMP TABLE quality (source_row integer, quality_status text, "
                "quality_reasons text[]) ON COMMIT DROP"
            )
            cur.copy_expert("COPY quality FROM STDIN WITH CSV", _csv_buffer(statuses))
            cur.execute(
                "UPDATE raw.properties AS p SET quality_status = q.quality_status, "
                "quality_reasons = q.quality_reasons FROM quality AS q "
                "WHERE p.ingestion_id = %s AND p.source_row = q.source_row",
                (ingestion_id,),
            )

    # ------------------------------------------------------------- clean.properties
    def insert_clean(self, frame: pd.DataFrame) -> int:
        columns = ", ".join(CLEAN_COLUMNS)
        rows = frame[list(CLEAN_COLUMNS)].itertuples(index=False, name=None)
        with self.cursor() as cur:
            cur.execute(
                "CREATE TEMP TABLE incoming (LIKE clean.properties INCLUDING DEFAULTS) "
                "ON COMMIT DROP"
            )
            cur.copy_expert(f"COPY incoming ({columns}) FROM STDIN WITH CSV", _csv_buffer(rows))
            cur.execute(
                f"INSERT INTO clean.properties ({columns}) SELECT {columns} FROM incoming "
                "ON CONFLICT (row_hash) DO NOTHING"
            )
            return cur.rowcount

    def distinct_values(self, column: str, before_ingestion: int) -> list:
        if column not in CATEGORY_COLUMNS:
            raise ValueError(f"Columna no permitida: {column}")
        with self.cursor() as cur:
            cur.execute(
                f"SELECT DISTINCT {column} FROM clean.properties WHERE ingestion_id < %s",
                (before_ingestion,),
            )
            return [row[0] for row in cur.fetchall()]

    def count_clean(self, before_ingestion: int | None = None, upto_ingestion: int | None = None):
        if before_ingestion is not None:
            return self.scalar(
                "SELECT count(*) FROM clean.properties WHERE ingestion_id < %s", (before_ingestion,)
            )
        if upto_ingestion is not None:
            return self.scalar(
                "SELECT count(*) FROM clean.properties WHERE ingestion_id <= %s", (upto_ingestion,)
            )
        return self.scalar("SELECT count(*) FROM clean.properties")

    def clean_ingestions(self) -> list[int]:
        with self.cursor() as cur:
            cur.execute("SELECT DISTINCT ingestion_id FROM clean.properties ORDER BY 1")
            return [row[0] for row in cur.fetchall()]

    # ------------------------------------------------------------- audit
    def start_audit(self, ingestion_id: int, **fields) -> None:
        columns = ["ingestion_id", *fields]
        values = [ingestion_id, *fields.values()]
        updates = ", ".join(f"{c} = EXCLUDED.{c}" for c in fields)
        with self.cursor() as cur:
            cur.execute(
                f"INSERT INTO audit.batch_runs ({', '.join(columns)}) "
                f"VALUES ({', '.join(['%s'] * len(columns))}) "
                f"ON CONFLICT (ingestion_id) DO UPDATE SET {updates}, started_at = now(), "
                "final_status = 'running', error = NULL, finished_at = NULL",
                values,
            )

    def update_audit(self, ingestion_id: int, **fields) -> None:
        clause, values = self._set_clause(fields)
        with self.cursor() as cur:
            cur.execute(
                f"UPDATE audit.batch_runs SET {clause} WHERE ingestion_id = %s",
                [*values, ingestion_id],
            )

    def audit(self, ingestion_id: int) -> dict | None:
        return self.one("SELECT * FROM audit.batch_runs WHERE ingestion_id = %s", (ingestion_id,))

    def last_training(self, before_ingestion: int) -> dict | None:
        return self.one(
            "SELECT ingestion_id, finished_at FROM audit.batch_runs "
            "WHERE training_decision = 'train' AND candidate_version IS NOT NULL "
            "AND ingestion_id < %s ORDER BY ingestion_id DESC LIMIT 1",
            (before_ingestion,),
        )

    def add_evaluations(self, rows: list[dict]) -> None:
        if not rows:
            return
        columns = ("ingestion_id", "cycle", "model_version", "role", "dataset", "metric", "value")
        columns += ("n_rows",)
        with self.cursor() as cur:
            cur.executemany(
                f"INSERT INTO audit.model_evaluations ({', '.join(columns)}) "
                f"VALUES ({', '.join(['%s'] * len(columns))})",
                [tuple(_cell(row.get(c)) for c in columns) for row in rows],
            )

    def inference_requests(self, since, sources: tuple[str, ...]) -> pd.DataFrame:
        rows = self.all(
            "SELECT request FROM raw.inference_events WHERE status = 'ok' "
            "AND traffic_source IN %s AND ts > %s",
            (sources, since),
        )
        return pd.DataFrame([row["request"] for row in rows if row["request"]])
