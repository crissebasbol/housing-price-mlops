import json
import logging
import re
import tempfile
import time
from dataclasses import asdict
from datetime import UTC, date, datetime
from pathlib import Path

import mlflow
import pandas as pd
import requests

import housing_ml
from housing_ml.cleaning import RANGES, normalize, quality_flags, quality_report, rejection_reasons
from housing_ml.decision import BatchSignals, TrainingThresholds, decide_training
from housing_ml.drift import DRIFT_COLUMNS, category_psi, new_categories, numeric_drift, psi
from housing_ml.features import MODEL_INPUT_COLUMNS, assign_splits, payload_hash, row_hash
from housing_ml.formatting import percent
from housing_ml.gates import EvaluationSet, PromotionThresholds, compare_models
from housing_ml.metrics import price_band, regression_metrics, size_band
from housing_ml.model import build_model, describe_model, iterations
from housing_ml.schema import received_schema
from housing_ml.schema import validate_schema as check_schema
from pipeline import registry
from pipeline.config import (
    DATA_API_DAY,
    DATA_API_GROUP,
    DATA_API_TIMEOUT,
    DATA_API_URL,
    DATASETS_BUCKET,
    END_OF_DATA_POLICY,
    GIT_SHA,
    INFERENCE_SOURCES_FOR_DRIFT,
    LANDING_BUCKET,
    MAX_RECOVERY_ATTEMPTS,
    MLFLOW_TRACKING_URI,
    REPLAY_OBJECT,
    REPLAY_SAMPLE_ROWS,
    SHAP_ROWS,
)
from pipeline.data_api import DataApiClient
from pipeline.database import CATEGORY_COLUMNS, Database, _json_default, as_json
from pipeline.reports import prediction_plots, shap_plots
from pipeline.storage import Storage, work_key

log = logging.getLogger(__name__)

FETCH_TASK = "fetch_batch_from_api"
PREPROCESSING_VERSION = f"housing_ml-{housing_ml.__version__}+{GIT_SHA[:7]}"
INPUTS = list(MODEL_INPUT_COLUMNS)
CLEAN_SELECT = "SELECT ingestion_id, price, " + ", ".join(INPUTS) + " FROM clean.properties"
METRICS = ("mape", "mdape", "mae", "rmse", "r2")
MIN_INFERENCE_EVENTS = TrainingThresholds().min_inference_events


def _client() -> DataApiClient:
    return DataApiClient(DATA_API_URL, DATA_API_GROUP, DATA_API_DAY, DATA_API_TIMEOUT)


def _batch(ti) -> dict:
    return ti.xcom_pull(task_ids=FETCH_TASK)


def _jsonable(value):
    return json.loads(json.dumps(value, default=_json_default))


def _summary(row: dict, recovered: bool = False) -> dict:
    keys = (
        "ingestion_id",
        "outcome",
        "cycle",
        "api_batch_number",
        "expected_batch_number",
        "record_count",
        "landing_uri",
        "duplicate_of",
        "data_sha256",
    )
    return {**{k: row.get(k) for k in keys}, "recovered": recovered}


def _records(storage: Storage, batch: dict) -> list[dict]:
    return storage.get_json_gz(batch["landing_uri"])["data"]


def _valid_rows(storage: Storage, ingestion_id: int) -> pd.DataFrame:
    frame = storage.get_parquet(work_key(ingestion_id, "normalized.parquet"))
    return frame[frame["valid"]].reset_index(drop=True)


def _evaluation_rows(batch: dict, version, role: str, dataset: str, metrics: dict) -> list[dict]:
    return [
        {
            "ingestion_id": batch["ingestion_id"],
            "cycle": batch["cycle"],
            "model_version": version,
            "role": role,
            "dataset": dataset,
            "metric": name,
            "value": metrics[name],
            "n_rows": metrics.get("n"),
        }
        for name in METRICS
        if metrics.get(name) is not None
    ]


# --------------------------------------------------------------------- ingesta
def check_dependencies(**_) -> None:
    def mlflow_health():
        requests.get(f"{MLFLOW_TRACKING_URI}/health", timeout=10).raise_for_status()

    checks = {
        "postgres": Database().check,
        "minio": Storage().check,
        "mlflow": mlflow_health,
        "api de datos": _client().health,
    }
    problems = []
    for name, check in checks.items():
        try:
            check()
            log.info("%s responde", name)
        except Exception as error:
            problems.append(f"{name}: {error}")
    if problems:
        raise RuntimeError(
            "No se pide ningún lote porque faltan dependencias: " + "; ".join(problems)
        )


def fetch_batch_from_api(run_id, **_) -> dict:
    db, storage = Database(), Storage()

    same_run = db.batch_for_run(run_id)
    if same_run:
        log.info(
            "Reintento de la misma ejecución: se reutiliza el ingreso %s", same_run["ingestion_id"]
        )
        return _summary(same_run)

    pending = db.recoverable_batch(MAX_RECOVERY_ATTEMPTS)
    if pending:
        db.update_batch(
            pending["ingestion_id"],
            dag_run_id=run_id,
            attempts=pending["attempts"] + 1,
            status="fetched",
            error=None,
        )
        db.start_audit(
            pending["ingestion_id"],
            dag_run_id=run_id,
            cycle=pending["cycle"],
            api_batch_number=pending["api_batch_number"],
            outcome="new",
            record_count=pending["record_count"],
        )
        log.warning(
            "Se reprocesa el ingreso %s desde el landing (intento %s): "
            "la ejecución anterior no terminó",
            pending["ingestion_id"],
            pending["attempts"] + 1,
        )
        return _summary(db.batch(pending["ingestion_id"]), recovered=True)

    if quarantined := db.quarantine_exhausted(MAX_RECOVERY_ATTEMPTS):
        log.warning(
            "%s ingreso(s) en cuarentena tras %s intentos", quarantined, MAX_RECOVERY_ATTEMPTS
        )

    last = db.last_batch()
    cycle = 1
    if last:
        restarted = last["outcome"] == "end_of_data" and END_OF_DATA_POLICY == "restart"
        cycle = last["cycle"] + (1 if restarted else 0)
    expected = db.next_expected_batch(cycle)

    response = _client().fetch()
    fields = {
        "dag_run_id": run_id,
        "group_number": DATA_API_GROUP,
        "day": DATA_API_DAY,
        "http_status": response.status_code,
        "record_count": len(response.records),
    }
    if response.outcome == "data":
        number = response.batch_number
        if number is not None and number < expected:
            log.warning(
                "La API entregó el lote %s y se esperaba el %s: se reinició por fuera; "
                "empieza el ciclo %s",
                number,
                expected,
                cycle + 1,
            )
            cycle, expected = cycle + 1, 0
        elif number is not None and number > expected:
            log.warning(
                "Lote fuera de secuencia: llegó el %s y se esperaba el %s", number, expected
            )
        data_sha256 = payload_hash(response.records)
        original = db.find_new_batch(data_sha256)
        if original:
            fields.update(outcome="duplicate", duplicate_of=original, status="closed")
        else:
            key = f"cycle={cycle}/batch={number}/{re.sub(r'[^A-Za-z0-9_.-]', '_', run_id)}.json.gz"
            payload = {
                "group_number": DATA_API_GROUP,
                "day": DATA_API_DAY,
                "batch_number": number,
                "data": response.records,
            }
            landing = storage.put_json_gz(LANDING_BUCKET, key, payload)
            fields.update(outcome="new", landing_uri=landing, status="fetched")
        fields.update(api_batch_number=number, data_sha256=data_sha256)
    else:
        fields.update(outcome=response.outcome, status="closed")
    fields.update(cycle=cycle, expected_batch_number=expected)

    ingestion_id = db.insert_batch(**fields)
    db.start_audit(
        ingestion_id,
        dag_run_id=run_id,
        cycle=cycle,
        api_batch_number=fields.get("api_batch_number"),
        outcome=fields["outcome"],
        record_count=fields["record_count"],
    )
    log.info(
        "Ingreso %s: %s, ciclo %s, lote %s, %s filas",
        ingestion_id,
        fields["outcome"],
        cycle,
        fields.get("api_batch_number"),
        fields["record_count"],
    )
    return _summary(db.batch(ingestion_id))


def store_raw_batch(ti, **_) -> None:
    batch = _batch(ti)
    if batch["outcome"] != "new":
        log.info("Sin filas que guardar en RAW (%s)", batch["outcome"])
        return
    db, storage = Database(), Storage()
    records = _records(storage, batch)
    hashes = [row_hash(record) for record in records]
    db.replace_raw_properties(batch["ingestion_id"], records, hashes)
    schema = received_schema(pd.DataFrame(records))
    db.update_batch(batch["ingestion_id"], status="stored", schema_received=schema)
    log.info(
        "Ingreso %s: %s filas originales en raw.properties", batch["ingestion_id"], len(records)
    )


def route_batch(ti, **_) -> str:
    outcome = _batch(ti)["outcome"]
    return {
        "new": "validate_schema",
        "duplicate": "monitor_production_model",
        "end_of_data": "handle_end_of_data",
        "empty": "record_empty_batch",
    }[outcome]


# --------------------------------------------------------------------- validación
def validate_schema(ti, **_) -> str:
    batch = _batch(ti)
    report = check_schema(pd.DataFrame(_records(Storage(), batch)))
    Database().update_audit(
        batch["ingestion_id"], schema_status=report.status, schema_report=report.to_dict()
    )
    log.info("Esquema %s: faltan %s, nuevas %s", report.status, report.missing, report.extra)
    return report.status


def validate_data_quality(ti, **_) -> dict:
    batch = _batch(ti)
    db, storage = Database(), Storage()
    records = _records(storage, batch)
    raw = pd.DataFrame(records)
    clean = normalize(raw, date.today())
    flags = quality_flags(clean)
    report = quality_report(raw, clean, flags)
    reasons = rejection_reasons(flags).to_dict()
    db.set_quality(
        batch["ingestion_id"],
        [
            (position, "rejected" if position in reasons else "valid", reasons.get(position))
            for position in range(len(raw))
        ],
    )
    normalized = clean.assign(
        row_hash=[row_hash(record) for record in records],
        valid=~flags.any(axis=1).to_numpy(),
    )
    storage.put_parquet(work_key(batch["ingestion_id"], "normalized.parquet"), normalized)
    db.update_audit(
        batch["ingestion_id"],
        record_count=report["rows"],
        valid_rows=report["valid_rows"],
        rejected_rows=report["rejected_rows"],
        quality_report=report,
    )
    log.info(
        "Calidad: %s válidas, %s rechazadas %s",
        report["valid_rows"],
        report["rejected_rows"],
        report["rejections_by_reason"],
    )
    return {"valid_rows": report["valid_rows"], "rejected_rows": report["rejected_rows"]}


def detect_new_categories(ti, **_) -> dict:
    batch = _batch(ti)
    db = Database()
    valid = _valid_rows(Storage(), batch["ingestion_id"])
    result = {}
    if db.count_clean(before_ingestion=batch["ingestion_id"]):
        for column in CATEGORY_COLUMNS:
            seen = pd.Series(
                db.distinct_values(column, before_ingestion=batch["ingestion_id"]), dtype=object
            )
            result[column] = new_categories(seen, valid[column])
    db.update_audit(batch["ingestion_id"], new_categories=result)
    shares = {column: value["row_share"] for column, value in result.items()}
    log.info("Categorías nuevas (porcentaje de filas del lote): %s", shares)
    return shares


def detect_data_drift(ti, **_) -> dict:
    batch = _batch(ti)
    db = Database()
    valid = _valid_rows(Storage(), batch["ingestion_id"])
    reference = db.read_frame(
        "SELECT price, house_size, bed, bath, acre_lot, status, state FROM clean.properties "
        "WHERE ingestion_id < %s",
        (batch["ingestion_id"],),
    )
    report = {}
    if not reference.empty:
        report = numeric_drift(reference, valid)
        for column in ("status", "state"):
            value = category_psi(reference[column], valid[column])
            report[column] = {"psi": None if value is None else round(value, 4)}
    db.update_audit(batch["ingestion_id"], drift_report=report)
    summary = {column: values["psi"] for column, values in report.items()}
    log.info("PSI frente al acumulado: %s", summary)
    return summary


def preprocess_data(ti, **_) -> dict:
    batch = _batch(ti)
    valid = _valid_rows(Storage(), batch["ingestion_id"])
    rows = valid.assign(
        ingestion_id=batch["ingestion_id"],
        cycle=batch["cycle"],
        api_batch_number=batch["api_batch_number"],
        split=assign_splits(valid["row_hash"]),
        preprocessing_version=PREPROCESSING_VERSION,
    )
    inserted = Database().insert_clean(rows) if len(rows) else 0
    log.info("CLEAN: %s filas nuevas de %s válidas", inserted, len(valid))
    return {"inserted": inserted, "valid": len(valid)}


def export_replay_sample(**_) -> None:
    sample = Database().read_frame(
        "SELECT row_hash, price, " + ", ".join(INPUTS) + " FROM clean.properties "
        "WHERE split = 'holdout' ORDER BY md5(row_hash) LIMIT %s",
        (REPLAY_SAMPLE_ROWS,),
    )
    sample["prev_sold_date"] = sample["prev_sold_date"].dt.strftime("%Y-%m-%d")
    records = sample.astype(object).where(sample.notna(), None).to_dict("records")
    uri = Storage().put_json(DATASETS_BUCKET, REPLAY_OBJECT, records)
    log.info("Muestra de replay: %s filas holdout en %s", len(records), uri)


# --------------------------------------------------------------------- decisión
def evaluate_production_on_batch(ti, **_) -> dict | None:
    batch = _batch(ti)
    db = Database()
    current = registry.champion()
    if current is None:
        log.info("Todavía no hay modelo productivo")
        return None
    model = registry.load_version(current["version"])
    valid = _valid_rows(Storage(), batch["ingestion_id"])
    metrics = regression_metrics(valid["price"], model.predict(valid[INPUTS]))
    db.update_audit(
        batch["ingestion_id"],
        champion_version_before=current["version"],
        champion_mape_on_batch=metrics["mape"],
    )
    db.add_evaluations(
        _evaluation_rows(batch, current["version"], "champion", "new_batch", metrics)
    )
    log.info(
        "Modelo productivo v%s en el lote nuevo: MAPE %s (referencia %s)",
        current["version"],
        percent(metrics["mape"], 2),
        current["reference_mape"],
    )
    return {
        "version": current["version"],
        "mape": metrics["mape"],
        "reference_mape": current["reference_mape"],
    }


def _inference_drift(db: Database, last_training: dict | None) -> tuple[dict, int]:
    since = last_training and last_training.get("finished_at")
    if since is None:
        return {}, 0
    requests_frame = db.inference_requests(since, INFERENCE_SOURCES_FOR_DRIFT)
    if len(requests_frame) < MIN_INFERENCE_EVENTS:
        return {}, len(requests_frame)
    reference = db.read_frame(
        "SELECT house_size, bed, bath, acre_lot FROM clean.properties WHERE split = 'train' "
        "ORDER BY md5(row_hash) LIMIT 200000"
    )
    result = {}
    for column in ("house_size", "bed", "bath", "acre_lot"):
        if column in requests_frame:
            current = pd.to_numeric(requests_frame[column], errors="coerce")
            result[column] = psi(reference[column], current)
    return result, len(requests_frame)


def decide_training_branch(ti, **_) -> str:
    batch = _batch(ti)
    db = Database()
    audit = db.audit(batch["ingestion_id"])
    evaluation = ti.xcom_pull(task_ids="evaluate_production_on_batch")
    last = db.last_training(batch["ingestion_id"])
    inference_psi, inference_events = _inference_drift(db, last)
    signals = BatchSignals(
        schema_status=audit["schema_status"],
        valid_rows=audit["valid_rows"] or 0,
        rejected_rows=audit["rejected_rows"] or 0,
        accumulated_valid_rows=db.count_clean(before_ingestion=batch["ingestion_id"]),
        rows_at_last_training=(
            db.count_clean(upto_ingestion=last["ingestion_id"]) if last else None
        ),
        has_champion=evaluation is not None,
        drift_psi={
            column: values.get("psi")
            for column, values in (audit["drift_report"] or {}).items()
            if column in DRIFT_COLUMNS
        },
        new_category_share={
            column: values["row_share"]
            for column, values in (audit["new_categories"] or {}).items()
        },
        champion_mape_on_batch=evaluation["mape"] if evaluation else None,
        champion_reference_mape=evaluation["reference_mape"] if evaluation else None,
        inference_psi=inference_psi,
        inference_events=inference_events,
    )
    decision = decide_training(signals)
    db.update_audit(
        batch["ingestion_id"],
        training_decision="train" if decision.train else "skip",
        training_code=decision.code,
        training_signals=decision.signals,
        training_reason=decision.reason,
    )
    log.info(decision.reason)
    return "train_candidate_model" if decision.train else "skip_training"


def skip_training(ti, **_) -> None:
    audit = Database().audit(_batch(ti)["ingestion_id"])
    log.info("No se entrena (%s): %s", audit["training_code"], audit["training_reason"])


# --------------------------------------------------------------------- entrenamiento
def train_candidate_model(ti, run_id, **_) -> dict:
    batch = _batch(ti)
    db = Database()
    audit = db.audit(batch["ingestion_id"])
    train = db.read_frame(CLEAN_SELECT + " WHERE split = 'train'")
    model = build_model()
    started = time.perf_counter()
    model.fit(train[INPUTS], train["price"])
    seconds = time.perf_counter() - started

    registry.configure()
    tags = {
        "ingestion_id": batch["ingestion_id"],
        "api_batch_number": batch["api_batch_number"],
        "cycle": batch["cycle"],
        "batches_used": ",".join(str(i) for i in sorted(train["ingestion_id"].unique())),
        "data_sha256": batch["data_sha256"],
        "train_rows": len(train),
        "git_sha": GIT_SHA,
        "preprocessing_version": PREPROCESSING_VERSION,
        "dag_run_id": run_id,
        "training_reason": audit["training_reason"],
        "training_signals": ",".join(audit["training_signals"] or []),
    }
    run_name = f"c{batch['cycle']}-b{batch['api_batch_number']}-{datetime.now(UTC):%Y%m%d%H%M}"
    with mlflow.start_run(run_name=run_name, tags={k: str(v) for k, v in tags.items()}) as run:
        mlflow.log_params(describe_model())
        mlflow.log_params({f"decision_{k}": v for k, v in asdict(TrainingThresholds()).items()})
        mlflow.log_params({f"promotion_{k}": v for k, v in asdict(PromotionThresholds()).items()})
        mlflow.log_params({f"quality_{k}": f"{low}-{high}" for k, (low, high) in RANGES.items()})
        metrics = regression_metrics(train["price"], model.predict(train[INPUTS]))
        mlflow.log_metrics({f"train_{k}": metrics[k] for k in METRICS})
        mlflow.log_metrics(
            {"train_rows": len(train), "training_seconds": seconds, "iterations": iterations(model)}
        )
        info = registry.log_model(
            model, train[INPUTS].sample(min(200, len(train)), random_state=42)
        )
    log.info(
        "Candidato entrenado en %.0f s con %s filas (run %s)", seconds, len(train), run.info.run_id
    )
    return {"run_id": run.info.run_id, "model_uri": info.model_uri, "train_rows": len(train)}


def evaluate_candidate_model(ti, **_) -> dict:
    batch = _batch(ti)
    trained = ti.xcom_pull(task_ids="train_candidate_model")
    db = Database()
    model = registry.load_uri(trained["model_uri"])
    results = {}
    with tempfile.TemporaryDirectory() as tmp, mlflow.start_run(run_id=trained["run_id"]):
        out = Path(tmp)
        for split in ("val", "test"):
            frame = db.read_frame(CLEAN_SELECT + " WHERE split = %s", (split,))
            predictions = model.predict(frame[INPUTS])
            metrics = regression_metrics(frame["price"], predictions)
            mlflow.log_metrics({f"{split}_{k}": metrics[k] for k in METRICS})
            mlflow.log_metric(f"{split}_rows", metrics["n"])
            results[f"{split}_mape"] = metrics["mape"]
            if split == "val":
                for path in prediction_plots(frame["price"], predictions, out, "val"):
                    mlflow.log_artifact(str(path), "plots")
                sample = frame[INPUTS].sample(min(SHAP_ROWS, len(frame)), random_state=42)
                paths, importance = shap_plots(model, sample, out)
                for path in paths:
                    mlflow.log_artifact(str(path), "shap")
                if importance:
                    mlflow.log_dict(importance, "shap/importancia_media.json")
        audit = db.audit(batch["ingestion_id"])
        mlflow.log_dict(
            _jsonable(
                {
                    "schema": audit["schema_report"],
                    "quality": audit["quality_report"],
                    "new_categories": audit["new_categories"],
                }
            ),
            "data/validation_report.json",
        )
        mlflow.log_dict(_jsonable(audit["drift_report"] or {}), "data/drift_report.json")
    log.info(
        "Validación %s, prueba %s",
        percent(results["val_mape"], 2),
        percent(results["test_mape"], 2),
    )
    return results


def register_candidate_in_mlflow(ti, **_) -> dict:
    batch = _batch(ti)
    trained = ti.xcom_pull(task_ids="train_candidate_model")
    db = Database()
    audit = db.audit(batch["ingestion_id"])
    version = registry.register(
        trained["model_uri"],
        {
            "ingestion_id": batch["ingestion_id"],
            "cycle": batch["cycle"],
            "api_batch_number": batch["api_batch_number"],
            "git_sha": GIT_SHA,
            "training_reason": audit["training_reason"],
        },
    )
    db.update_audit(
        batch["ingestion_id"], mlflow_run_id=trained["run_id"], candidate_version=version
    )
    log.info("Candidato registrado como versión %s (alias challenger)", version)
    return {"version": version}


# --------------------------------------------------------------------- promoción
def compare_with_production(ti, **_) -> dict:
    batch = _batch(ti)
    trained = ti.xcom_pull(task_ids="train_candidate_model")
    candidate_version = ti.xcom_pull(task_ids="register_candidate_in_mlflow")["version"]
    db = Database()
    holdout = db.read_frame(CLEAN_SELECT + " WHERE split = 'holdout'")
    last = db.last_training(batch["ingestion_id"])
    since = last["ingestion_id"] if last else 0
    evaluation = EvaluationSet(
        X=holdout[INPUTS],
        y=holdout["price"].to_numpy(),
        recent=(holdout["ingestion_id"] > since).to_numpy(),
        segments={
            "status": holdout["status"],
            "state": holdout["state"],
            "size_band": size_band(holdout["house_size"]),
        },
        diagnostic_segments={"price_band": price_band(holdout["price"])},
    )
    candidate = registry.load_version(candidate_version)
    current = registry.champion()
    champion_model = registry.load_version(current["version"]) if current else None
    decision = compare_models(candidate, champion_model, evaluation)
    payload = _jsonable(decision.to_dict())

    with mlflow.start_run(run_id=trained["run_id"]):
        mlflow.log_metrics({f"holdout_{k}": decision.candidate_metrics[k] for k in METRICS})
        if decision.champion_metrics:
            mlflow.log_metrics(
                {f"champion_holdout_{k}": decision.champion_metrics[k] for k in METRICS}
            )
        mlflow.log_metrics({f"gate_{g.gate}": int(g.passed) for g in decision.gates})
        budget = decision.gate("G6_budget")
        if budget:
            mlflow.log_metrics(
                {
                    "latency_p95_ms": budget.detail["latency_p95_ms"],
                    "model_size_mb": budget.detail["size_mb"],
                }
            )
        mlflow.log_dict(payload, "promotion/gates.json")
        mlflow.set_tag("promotion_decision", "promoted" if decision.promote else "rejected")
        mlflow.set_tag("promotion_reason", decision.reason)

    rows = _evaluation_rows(
        batch, candidate_version, "candidate", "holdout_total", decision.candidate_metrics
    )
    if current and decision.champion_metrics:
        rows += _evaluation_rows(
            batch, current["version"], "champion", "holdout_total", decision.champion_metrics
        )
    segments = decision.gate("G4_segments")
    for item in (segments.detail["segments"] if segments else []) + decision.diagnostics:
        rows.append(
            {
                "ingestion_id": batch["ingestion_id"],
                "cycle": batch["cycle"],
                "model_version": candidate_version,
                "role": "candidate",
                "dataset": f"segment:{item['segment']}",
                "metric": "mape",
                "value": item["candidate"],
                "n_rows": item["rows"],
            }
        )
    db.add_evaluations(rows)

    holdout_gate = decision.gate("G1_holdout")
    db.update_audit(
        batch["ingestion_id"],
        gates=as_json(payload["gates"]),
        diagnostics=as_json(payload["diagnostics"]),
        mape_candidate=decision.candidate_metrics["mape"],
        mape_champion=decision.champion_metrics["mape"] if decision.champion_metrics else None,
        mape_delta_pct=holdout_gate.detail["mape_change"] * 100 if holdout_gate else None,
        mae_delta_pct=holdout_gate.detail["mae_change"] * 100 if holdout_gate else None,
        r2_delta=holdout_gate.detail["r2_change"] if holdout_gate else None,
    )
    log.info(decision.reason)
    return {
        "promote": decision.promote,
        "reason": decision.reason,
        "candidate_version": candidate_version,
        "champion_version": current["version"] if current else None,
        "holdout_mape": decision.candidate_metrics["mape"],
    }


def decide_promotion_branch(ti, **_) -> str:
    result = ti.xcom_pull(task_ids="compare_with_production")
    return "promote_model" if result["promote"] else "reject_model"


def promote_model(ti, **_) -> None:
    batch = _batch(ti)
    result = ti.xcom_pull(task_ids="compare_with_production")
    previous = registry.promote(result["candidate_version"])
    registry.tag_version(
        result["candidate_version"],
        {
            "decision": "promoted",
            "reason": result["reason"],
            "holdout_mape": result["holdout_mape"],
        },
    )
    Database().update_audit(
        batch["ingestion_id"],
        promotion_decision="promoted",
        promotion_reason=result["reason"],
        champion_version_after=result["candidate_version"],
    )
    log.info("Versión %s ahora es champion (antes %s)", result["candidate_version"], previous)


def reject_model(ti, **_) -> None:
    batch = _batch(ti)
    result = ti.xcom_pull(task_ids="compare_with_production")
    registry.tag_version(
        result["candidate_version"], {"decision": "rejected", "reason": result["reason"]}
    )
    Database().update_audit(
        batch["ingestion_id"],
        promotion_decision="rejected",
        promotion_reason=result["reason"],
        champion_version_after=result["champion_version"],
    )
    log.info("Versión %s rechazada: %s", result["candidate_version"], result["reason"])


# --------------------------------------------------------------------- otros caminos
def monitor_production_model(ti, **_) -> None:
    batch = _batch(ti)
    db = Database()
    reason = (
        f"Lote repetido: su contenido es idéntico al ingreso {batch['duplicate_of']} "
        "(misma huella SHA-256). No se reingiere ni se reentrena; se usa para vigilar al "
        "modelo productivo."
    )
    fields = {"training_decision": "skip", "training_code": "duplicate"}
    current = registry.champion()
    if current:
        holdout = db.read_frame(
            CLEAN_SELECT + " WHERE ingestion_id = %s AND split = 'holdout'",
            (batch["duplicate_of"],),
        )
        if len(holdout):
            model = registry.load_version(current["version"])
            metrics = regression_metrics(holdout["price"], model.predict(holdout[INPUTS]))
            db.add_evaluations(
                _evaluation_rows(batch, current["version"], "champion", "monitoring", metrics)
            )
            fields.update(
                champion_version_before=current["version"], champion_mape_on_batch=metrics["mape"]
            )
            reason += (
                f" MAPE del modelo productivo (versión {current['version']}) en el holdout de "
                f"ese lote: {percent(metrics['mape'], 2)}."
            )
    db.update_audit(batch["ingestion_id"], training_reason=reason, **fields)
    log.info(reason)


def handle_end_of_data(ti, **_) -> None:
    batch = _batch(ti)
    if END_OF_DATA_POLICY == "restart":
        _client().restart()
        reason = (
            "Fin de datos: la API ya entregó todos sus lotes. Se reinició (política restart); "
            f"el próximo lote abre el ciclo {batch['cycle'] + 1} y se usará para monitoreo."
        )
    else:
        reason = "Fin de datos: la API ya entregó todos sus lotes (política stop)."
    Database().update_audit(
        batch["ingestion_id"],
        training_decision="skip",
        training_code="end_of_data",
        training_reason=reason,
    )
    log.info(reason)


def record_empty_batch(ti, **_) -> None:
    batch = _batch(ti)
    Database().update_audit(
        batch["ingestion_id"],
        training_decision="skip",
        training_code="empty",
        training_reason="La API respondió sin filas: no hay nada que validar ni entrenar.",
    )


def notify_or_log_result(ti, **_) -> None:
    batch = _batch(ti)
    db = Database()
    audit = db.audit(batch["ingestion_id"])
    if batch["outcome"] == "new":
        final_status = "success" if audit["training_decision"] == "train" else "skipped"
    elif batch["outcome"] == "duplicate":
        final_status = "duplicate"
    else:
        final_status = "no_data"
    db.update_audit(batch["ingestion_id"], final_status=final_status, finished_at=datetime.now(UTC))
    db.update_batch(
        batch["ingestion_id"], status="processed" if batch["outcome"] == "new" else "closed"
    )
    log.info(
        "Ingreso %s (ciclo %s, lote %s): %s. %s %s",
        batch["ingestion_id"],
        batch["cycle"],
        batch["api_batch_number"],
        final_status,
        audit["training_reason"] or "",
        audit["promotion_reason"] or "",
    )


def on_task_failure(context) -> None:
    try:
        ti = context["ti"]
        batch = ti.xcom_pull(task_ids=FETCH_TASK)
        if not batch:
            return
        error = f"{ti.task_id}: {context.get('exception')}"[:2000]
        db = Database()
        db.update_audit(
            batch["ingestion_id"], final_status="failed", error=error, finished_at=datetime.now(UTC)
        )
        if batch["outcome"] == "new":
            db.update_batch(batch["ingestion_id"], status="failed", error=error)
    except Exception:
        log.exception("No se pudo registrar la falla en la auditoría")
