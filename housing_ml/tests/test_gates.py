import numpy as np
import pandas as pd
import pytest

from housing_ml.features import MODEL_INPUT_COLUMNS
from housing_ml.gates import (
    EvaluationSet,
    PromotionThresholds,
    compare_models,
    gate_budget,
    gate_holdout,
    gate_segments,
)
from housing_ml.metrics import price_band


class ScaledModel:
    """Predice el precio real multiplicado por un error fijo por fila (para pruebas)."""

    def __init__(self, prices: pd.Series, errors: np.ndarray):
        self.lookup = dict(zip(prices.index, prices.to_numpy() * (1 + errors), strict=True))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.array([self.lookup.get(i, 300_000.0) for i in X.index])


@pytest.fixture
def evaluation(raw):
    frame = raw.reset_index(drop=True)
    return frame, EvaluationSet(
        X=frame[list(MODEL_INPUT_COLUMNS)],
        y=frame["price"].to_numpy(),
        recent=np.arange(len(frame)) >= len(frame) // 2,
        segments={"price_band": price_band(frame["price"]), "state": frame["state"]},
    )


def errors(rows: int, scale: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.choice([-1, 1], rows) * np.abs(rng.normal(scale, scale / 4, rows))


def test_holdout_rule_edges():
    t = PromotionThresholds()
    champion = {"mape": 0.20, "mae": 50_000, "r2": 0.80}
    better = {"mape": 0.197, "mae": 50_400, "r2": 0.795}
    too_small = {"mape": 0.199, "mae": 50_000, "r2": 0.80}
    worse_mae = {"mape": 0.18, "mae": 51_000, "r2": 0.80}
    assert gate_holdout(better, champion, t).passed
    assert not gate_holdout(too_small, champion, t).passed
    assert not gate_holdout(worse_mae, champion, t).passed


def test_candidate_clearly_better_is_promoted(evaluation):
    frame, data = evaluation
    champion = ScaledModel(frame["price"], errors(len(frame), 0.20, 1))
    candidate = ScaledModel(frame["price"], errors(len(frame), 0.10, 2))
    decision = compare_models(
        candidate,
        champion,
        data,
        PromotionThresholds(bootstrap_samples=200),
        latency_p95_ms=5,
        size_mb=10,
    )
    assert decision.promote, decision.reason
    assert decision.reason.startswith("Se promueve")


def test_equivalent_candidate_is_rejected(evaluation):
    frame, data = evaluation
    champion = ScaledModel(frame["price"], errors(len(frame), 0.15, 1))
    candidate = ScaledModel(frame["price"], errors(len(frame), 0.15, 2))
    decision = compare_models(
        candidate,
        champion,
        data,
        PromotionThresholds(bootstrap_samples=200),
        latency_p95_ms=5,
        size_mb=10,
    )
    assert not decision.promote
    assert not decision.gate("G2_significance").passed
    assert decision.reason.startswith("Se rechaza")


def test_segment_regression_blocks_promotion(evaluation):
    frame, _ = evaluation
    y = frame["price"].to_numpy()
    labels = pd.Series(np.where(np.arange(len(y)) % 2 == 0, "A", "B"))
    champion = y * 1.10
    candidate = np.where(labels == "A", y * 1.01, y * 1.20)
    result = gate_segments(
        y, candidate, champion, {"group": labels}, PromotionThresholds(segment_min_rows=100)
    )
    assert not result.passed
    assert "group=B" in result.message


def test_budget_gate():
    t = PromotionThresholds()
    assert gate_budget(12.0, 30.0, t).passed
    assert not gate_budget(80.0, 30.0, t).passed
    assert not gate_budget(12.0, 500.0, t).passed


def test_baseline_needs_behavior_and_budget(raw):
    from housing_ml.model import build_model

    frame = raw.reset_index(drop=True)
    model = build_model({"max_iter": 50}).fit(frame[list(MODEL_INPUT_COLUMNS)], frame["price"])
    data = EvaluationSet(
        X=frame[list(MODEL_INPUT_COLUMNS)],
        y=frame["price"].to_numpy(),
        recent=np.ones(len(frame), dtype=bool),
    )
    decision = compare_models(model, None, data, latency_p95_ms=5, size_mb=10)
    assert decision.promote
    assert [g.gate for g in decision.gates] == ["G5_behavior", "G6_budget"]
    assert decision.champion_metrics is None


def test_diagnostic_segments_do_not_block(evaluation):
    frame, data = evaluation
    y = frame["price"].to_numpy()
    labels = pd.Series(np.where(np.arange(len(y)) % 2 == 0, "A", "B"))
    champion = ScaledModel(frame["price"], np.where(labels == "A", 0.10, 0.30))
    candidate = ScaledModel(frame["price"], np.where(labels == "A", 0.13, 0.05))
    data.segments = {}
    data.diagnostic_segments = {"band": labels}
    decision = compare_models(
        candidate,
        champion,
        data,
        PromotionThresholds(bootstrap_samples=200, segment_min_rows=100),
        latency_p95_ms=5,
        size_mb=10,
    )
    assert decision.promote, decision.reason
    worse = [d for d in decision.diagnostics if d["segment"] == "band=A"][0]
    assert worse["candidate"] > worse["champion"] and not worse["passed"]
