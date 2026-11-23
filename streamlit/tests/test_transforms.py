import math

import pandas as pd
import pytest

from housing_ui import transforms


def url(version: int) -> str:
    return f"http://mlflow/#/models/m/versions/{version}"


@pytest.fixture
def runs() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ingestion_id": 1,
                "cycle": 1,
                "api_batch_number": 0,
                "outcome": "new",
                "final_status": "success",
                "valid_rows": 73635,
                "rejected_rows": 149,
                "training_decision": "train",
                "training_signals": ["baseline"],
                "training_reason": "Entrena: línea base",
                "candidate_version": 1,
                "mape_candidate": 0.2326,
                "mape_champion": None,
                "mape_delta_pct": None,
                "promotion_decision": "rejected",
                "promotion_reason": "Se rechaza: G6",
                "champion_version_before": None,
                "champion_mape_on_batch": None,
                "champion_version_after": None,
            },
            {
                "ingestion_id": 2,
                "cycle": 1,
                "api_batch_number": 1,
                "outcome": "new",
                "final_status": "success",
                "valid_rows": 94000,
                "rejected_rows": 200,
                "training_decision": "train",
                "training_signals": ["baseline"],
                "training_reason": "Entrena: línea base",
                "candidate_version": 2,
                "mape_candidate": 0.2466,
                "mape_champion": None,
                "mape_delta_pct": None,
                "promotion_decision": "promoted",
                "promotion_reason": "Se promueve",
                "champion_version_before": None,
                "champion_mape_on_batch": None,
                "champion_version_after": 2,
            },
            {
                "ingestion_id": 3,
                "cycle": 1,
                "api_batch_number": 2,
                "outcome": "new",
                "final_status": "success",
                "valid_rows": 230000,
                "rejected_rows": 300,
                "training_decision": "train",
                "training_signals": ["drift", "volume"],
                "training_reason": "Entrena: drift",
                "candidate_version": 3,
                "mape_candidate": 0.2414,
                "mape_champion": 0.2601,
                "mape_delta_pct": -7.19,
                "promotion_decision": "promoted",
                "promotion_reason": "Se promueve",
                "champion_version_before": 2,
                "champion_mape_on_batch": 0.2904,
                "champion_version_after": 3,
            },
            {
                "ingestion_id": 4,
                "cycle": 2,
                "api_batch_number": 0,
                "outcome": "duplicate",
                "final_status": "duplicate",
                "valid_rows": None,
                "rejected_rows": None,
                "training_decision": None,
                "training_signals": None,
                "training_reason": "Lote repetido",
                "candidate_version": None,
                "mape_candidate": None,
                "mape_champion": None,
                "mape_delta_pct": None,
                "promotion_decision": None,
                "promotion_reason": None,
                "champion_version_before": 3,
                "champion_mape_on_batch": 0.3216,
                "champion_version_after": None,
            },
        ]
    )


def test_summary_counts_and_current_champion(runs):
    assert transforms.summary(runs) == {
        "batches": 4,
        "trainings": 3,
        "promotions": 2,
        "champion": 3,
    }


def test_summary_without_champion(runs):
    assert transforms.summary(runs.iloc[:1])["champion"] is None


def test_batch_table_labels_percentages_and_links(runs):
    table = transforms.batch_table(runs, url)
    first, third, fourth = table.iloc[0], table.iloc[2], table.iloc[3]
    assert first["Promoción"] == "rechazado"
    assert first["MAPE candidata %"] == pytest.approx(23.26)
    assert first["MLflow"] == url(1)
    assert third["Señales"] == "drift, volume"
    assert third["Cambio MAPE %"] == pytest.approx(-7.19)
    assert fourth["Llegada"] == "repetido"
    assert fourth["Estado"] == "monitoreo"
    assert fourth["MLflow"] is None


def test_champion_mape_series_includes_monitoring(runs):
    series = transforms.champion_mape_series(runs)
    assert series["Ingreso"].tolist() == [3, 4]
    assert series["Champion"].tolist() == ["v2", "v3"]
    assert series["Llegada"].tolist() == ["nuevo", "repetido"]
    assert series["MAPE %"].iloc[1] == pytest.approx(32.16)


def test_candidate_vs_champion_long_format(runs):
    duel = transforms.candidate_vs_champion(runs)
    assert len(duel) == 4
    third = duel[duel["Ingreso"] == 3].set_index("Modelo")["MAPE %"]
    assert third["candidata"] == pytest.approx(24.14)
    assert third["champion"] == pytest.approx(26.01)


def test_waterfall_reaches_the_prediction():
    base = 250_000.0
    shap = [0.30, -0.10, 0.05, -0.02, 0.01, 0.004]
    explanation = {
        "base_price": base,
        "predicted_price": math.expm1(math.log1p(base) + sum(shap)),
        "contributions": [
            {"feature": f"f{i}", "value": i, "shap_log": v, "effect_pct": 0.0}
            for i, v in enumerate(shap)
        ],
    }
    steps = transforms.waterfall(explanation, top=3)
    assert steps["Paso"].iloc[0] == "Precio base"
    assert steps["Paso"].iloc[-2] == "Otras 3 variables"
    assert steps["Fin"].iloc[-1] == pytest.approx(explanation["predicted_price"])
    middle = steps.iloc[1:-1]
    assert middle["Inicio"].iloc[1:].values == pytest.approx(middle["Fin"].iloc[:-1].values)
    assert middle["Tipo"].tolist() == ["sube", "baja", "sube", "baja"]


def test_alias_rows_reads_tags():
    aliases = {"champion": 3, "previous_champion": 2}
    versions = {
        3: {"tags": {"holdout_mape": "0.2414", "decision": "promoted", "git_sha": "abcdef12"}},
        2: {"tags": {"holdout_mape": "0.2466", "decision": "promoted"}},
    }
    rows = transforms.alias_rows(aliases, versions, url)
    assert rows["Alias"].tolist() == ["champion", "previous_champion"]
    assert rows["MAPE holdout %"].tolist() == [pytest.approx(24.14), pytest.approx(24.66)]
    assert rows["Commit"].tolist() == ["abcdef1", None]
    assert rows["Decisión"].iloc[0] == "promovido"
