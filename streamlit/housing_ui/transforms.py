import math

import pandas as pd

OUTCOME_LABELS = {
    "new": "nuevo",
    "duplicate": "repetido",
    "end_of_data": "fin de datos",
    "empty": "vacío",
}
STATUS_LABELS = {
    "success": "éxito",
    "skipped": "sin entrenar",
    "duplicate": "monitoreo",
    "no_data": "sin datos",
    "failed": "falló",
    "running": "en curso",
}
PROMOTION_LABELS = {"promoted": "promovido", "rejected": "rechazado"}
FEATURE_LABELS = {
    "zip_code": "Código postal",
    "zip3": "Zona postal (3 dígitos)",
    "city_state": "Ciudad",
    "state": "Estado",
    "status": "Estado de venta",
    "bed": "Habitaciones",
    "bath": "Baños",
    "house_size": "Área (pies²)",
    "acre_lot": "Lote (acres)",
    "prev_sold_year": "Año de venta previa",
    "brokered_by": "Agencia",
}


def _pct(value) -> float | None:
    return None if value is None or pd.isna(value) else round(float(value) * 100, 2)


def current_champion(runs: pd.DataFrame) -> int | None:
    versions = runs["champion_version_after"].dropna()
    return int(versions.iloc[-1]) if len(versions) else None


def summary(runs: pd.DataFrame) -> dict:
    return {
        "batches": int(len(runs)),
        "trainings": int((runs["training_decision"] == "train").sum()),
        "promotions": int((runs["promotion_decision"] == "promoted").sum()),
        "champion": current_champion(runs),
    }


def default_cycles(cycles: list[int]) -> list[int]:
    return sorted({cycles[0], cycles[-1]}) if cycles else []


INTEGER_COLUMNS = (
    "Ingreso",
    "Ciclo",
    "Lote",
    "Filas válidas",
    "Rechazadas",
    "Candidata",
    "Champion después",
)


def batch_table(runs: pd.DataFrame, version_url) -> pd.DataFrame:
    def link(version):
        return None if version is None or pd.isna(version) else version_url(int(version))

    table = pd.DataFrame(
        {
            "Ingreso": runs["ingestion_id"],
            "Ciclo": runs["cycle"],
            "Lote": runs["api_batch_number"],
            "Llegada": runs["outcome"].map(OUTCOME_LABELS).fillna(runs["outcome"]),
            "Estado": runs["final_status"].map(STATUS_LABELS).fillna(runs["final_status"]),
            "Filas válidas": runs["valid_rows"],
            "Rechazadas": runs["rejected_rows"],
            "Señales": runs["training_signals"].map(
                lambda s: ", ".join(s) if isinstance(s, list) else None
            ),
            "Decisión de entrenar": runs["training_reason"],
            "Candidata": runs["candidate_version"],
            "MAPE candidata %": runs["mape_candidate"].map(_pct),
            "MAPE champion %": runs["mape_champion"].map(_pct),
            "Cambio MAPE %": runs["mape_delta_pct"].map(
                lambda v: None if v is None or pd.isna(v) else round(float(v), 2)
            ),
            "Promoción": runs["promotion_decision"].map(PROMOTION_LABELS),
            "Razón de la promoción": runs["promotion_reason"],
            "Champion después": runs["champion_version_after"],
            "MLflow": runs["candidate_version"].map(link),
        }
    )
    for column in INTEGER_COLUMNS:
        table[column] = pd.to_numeric(table[column], errors="coerce").astype("Int64")
    for column in ("MAPE candidata %", "MAPE champion %", "Cambio MAPE %"):
        table[column] = pd.to_numeric(table[column], errors="coerce").astype("Float64")
    return table


def champion_mape_series(runs: pd.DataFrame) -> pd.DataFrame:
    rows = runs.dropna(subset=["champion_mape_on_batch"])
    return pd.DataFrame(
        {
            "Ingreso": rows["ingestion_id"].astype(int),
            "MAPE %": rows["champion_mape_on_batch"].astype(float) * 100,
            "Champion": rows["champion_version_before"].map(lambda v: f"v{int(v)}"),
            "Llegada": rows["outcome"].map(OUTCOME_LABELS).fillna(rows["outcome"]),
        }
    )


def candidate_vs_champion(runs: pd.DataFrame) -> pd.DataFrame:
    rows = runs.dropna(subset=["mape_candidate"])
    long = []
    for _, row in rows.iterrows():
        decision = PROMOTION_LABELS.get(row["promotion_decision"], row["promotion_decision"])
        long.append(
            {
                "Ingreso": int(row["ingestion_id"]),
                "Modelo": "candidata",
                "MAPE %": float(row["mape_candidate"]) * 100,
                "Decisión": decision,
            }
        )
        if row["mape_champion"] is not None and not pd.isna(row["mape_champion"]):
            long.append(
                {
                    "Ingreso": int(row["ingestion_id"]),
                    "Modelo": "champion",
                    "MAPE %": float(row["mape_champion"]) * 100,
                    "Decisión": decision,
                }
            )
    return pd.DataFrame(long, columns=["Ingreso", "Modelo", "MAPE %", "Decisión"])


def waterfall(explanation: dict, top: int = 8) -> pd.DataFrame:
    contributions = sorted(
        explanation["contributions"], key=lambda c: abs(c["shap_log"]), reverse=True
    )
    shown, rest = contributions[:top], contributions[top:]
    steps = [
        (FEATURE_LABELS.get(c["feature"], c["feature"]), c["value"], c["shap_log"]) for c in shown
    ]
    if rest:
        steps.append((f"Otras {len(rest)} variables", None, sum(c["shap_log"] for c in rest)))

    level = math.log1p(explanation["base_price"])
    rows = [
        {
            "Paso": "Precio base",
            "Valor": None,
            "Inicio": 0.0,
            "Fin": explanation["base_price"],
            "Efecto": explanation["base_price"],
            "Tipo": "base",
        }
    ]
    for label, value, shap_log in steps:
        start = math.expm1(level)
        level += shap_log
        end = math.expm1(level)
        rows.append(
            {
                "Paso": label,
                "Valor": None if value is None else str(value),
                "Inicio": start,
                "Fin": end,
                "Efecto": end - start,
                "Tipo": "sube" if shap_log >= 0 else "baja",
            }
        )
    rows.append(
        {
            "Paso": "Predicción",
            "Valor": None,
            "Inicio": 0.0,
            "Fin": math.expm1(level),
            "Efecto": math.expm1(level),
            "Tipo": "total",
        }
    )
    return pd.DataFrame(rows)


def alias_rows(aliases: dict[str, int], versions: dict[int, dict], version_url) -> pd.DataFrame:
    rows = []
    for alias in ("champion", "challenger", "previous_champion"):
        number = aliases.get(alias)
        if number is None:
            continue
        tags = versions.get(number, {}).get("tags", {})
        mape = tags.get("holdout_mape")
        rows.append(
            {
                "Alias": alias,
                "Versión": number,
                "Ingreso": tags.get("ingestion_id"),
                "Lote": tags.get("api_batch_number"),
                "Ciclo": tags.get("cycle"),
                "MAPE holdout %": None if mape is None else round(float(mape) * 100, 2),
                "Decisión": PROMOTION_LABELS.get(tags.get("decision"), tags.get("decision")),
                "Commit": (tags.get("git_sha") or "")[:7] or None,
                "MLflow": version_url(number),
            }
        )
    return pd.DataFrame(rows)
