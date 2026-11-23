import pickle
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from housing_ml.formatting import decimal, percent, signed_percent
from housing_ml.metrics import ape, paired_bootstrap, regression_metrics

UNSEEN = "__unseen__"


@dataclass(frozen=True)
class PromotionThresholds:
    min_mape_improvement: float = 0.01
    max_mae_degradation: float = 0.01
    max_r2_drop: float = 0.01
    bootstrap_samples: int = 1_000
    bootstrap_max_rows: int = 50_000
    min_recent_rows: int = 100
    segment_min_rows: int = 1_000
    segment_tolerance: float = 0.05
    top_states: int = 10
    plausible_range: tuple[float, float] = (5_000, 50_000_000)
    max_out_of_range_share: float = 0.01
    behavior_rows: int = 500
    size_increase: float = 0.20
    latency_calls: int = 200
    max_latency_p95_ms: float = 150.0
    max_model_size_mb: float = 200.0


@dataclass
class EvaluationSet:
    X: pd.DataFrame
    y: np.ndarray
    recent: np.ndarray
    segments: dict[str, pd.Series] = field(default_factory=dict)
    diagnostic_segments: dict[str, pd.Series] = field(default_factory=dict)


@dataclass
class GateResult:
    gate: str
    passed: bool
    message: str
    detail: dict = field(default_factory=dict)


@dataclass
class PromotionDecision:
    promote: bool
    reason: str
    gates: list[GateResult]
    candidate_metrics: dict
    champion_metrics: dict | None
    diagnostics: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def gate(self, name: str) -> GateResult | None:
        return next((g for g in self.gates if g.gate == name), None)


def relative_change(new: float, old: float) -> float:
    return (new - old) / old if old else float("inf")


def gate_holdout(candidate: dict, champion: dict, t: PromotionThresholds) -> GateResult:
    mape = relative_change(candidate["mape"], champion["mape"])
    mae = relative_change(candidate["mae"], champion["mae"])
    r2 = candidate["r2"] - champion["r2"]
    checks = {
        "mape": mape <= -t.min_mape_improvement,
        "mae": mae <= t.max_mae_degradation,
        "r2": r2 >= -t.max_r2_drop,
    }
    message = (
        f"G1 holdout: MAPE {percent(candidate['mape'], 2)} frente a "
        f"{percent(champion['mape'], 2)} ({signed_percent(mape)}, se exige "
        f"-{percent(t.min_mape_improvement, 0)}); MAE {signed_percent(mae)} "
        f"(máximo +{percent(t.max_mae_degradation, 0)}); R² {decimal(r2, 3)} "
        f"(mínimo -{decimal(t.max_r2_drop)})"
    )
    detail = {"mape_change": mape, "mae_change": mae, "r2_change": r2, "checks": checks}
    return GateResult("G1_holdout", all(checks.values()), message, detail)


def gate_significance(
    ape_candidate: np.ndarray, ape_champion: np.ndarray, t: PromotionThresholds
) -> GateResult:
    ci = paired_bootstrap(
        ape_candidate - ape_champion,
        samples=t.bootstrap_samples,
        max_rows=t.bootstrap_max_rows,
    )
    passed = ci["high"] < 0
    message = (
        f"G2 significancia: diferencia media de APE {decimal(ci['mean'], 4)}, "
        f"IC 95 % [{decimal(ci['low'], 4)}, {decimal(ci['high'], 4)}]"
        + ("" if passed else " (incluye 0 o es positiva)")
    )
    return GateResult("G2_significance", passed, message, ci)


def gate_recent(
    y: np.ndarray,
    candidate: np.ndarray,
    champion: np.ndarray,
    recent: np.ndarray,
    t: PromotionThresholds,
) -> GateResult:
    rows = int(recent.sum())
    if rows < t.min_recent_rows:
        return GateResult(
            "G3_recent",
            True,
            f"G3 datos recientes: solo {rows} filas, no se evalúa",
            {"rows": rows},
        )
    cand = float(ape(y[recent], candidate[recent]).mean())
    champ = float(ape(y[recent], champion[recent]).mean())
    passed = cand <= champ
    message = (
        f"G3 datos recientes: MAPE {percent(cand, 2)} frente a {percent(champ, 2)} "
        f"en {rows} filas de los lotes nuevos"
    )
    return GateResult(
        "G3_recent", passed, message, {"rows": rows, "candidate": cand, "champion": champ}
    )


def segment_table(
    y: np.ndarray,
    candidate: np.ndarray,
    champion: np.ndarray | None,
    segments: dict[str, pd.Series],
    t: PromotionThresholds,
) -> list[dict]:
    table = []
    for name, labels in segments.items():
        labels = pd.Series(np.asarray(labels, dtype=object))
        counts = labels.value_counts()
        counts = counts[counts >= t.segment_min_rows]
        if name == "state":
            counts = counts.head(t.top_states)
        for value, rows in counts.items():
            mask = (labels == value).to_numpy()
            cand = float(ape(y[mask], candidate[mask]).mean())
            champ = None if champion is None else float(ape(y[mask], champion[mask]).mean())
            table.append(
                {
                    "segment": f"{name}={value}",
                    "rows": int(rows),
                    "candidate": cand,
                    "champion": champ,
                    "passed": champ is None or cand <= champ * (1 + t.segment_tolerance),
                }
            )
    return table


def gate_segments(
    y: np.ndarray,
    candidate: np.ndarray,
    champion: np.ndarray,
    segments: dict[str, pd.Series],
    t: PromotionThresholds,
) -> GateResult:
    evaluated = segment_table(y, candidate, champion, segments, t)
    failed = [item for item in evaluated if not item["passed"]]
    if failed:
        listed = ", ".join(
            f"{f['segment']} ({percent(f['candidate'], 2)} frente a {percent(f['champion'], 2)})"
            for f in failed
        )
        message = f"G4 segmentos: empeoran más de {percent(t.segment_tolerance, 0)} en {listed}"
    else:
        message = f"G4 segmentos: ninguno de los {len(evaluated)} segmentos empeora"
    return GateResult("G4_segments", not failed, message, {"segments": evaluated})


def gate_behavior(model, X: pd.DataFrame, t: PromotionThresholds, seed: int = 42) -> GateResult:
    sample = X.sample(min(len(X), t.behavior_rows), random_state=seed).reset_index(drop=True)
    predictions = np.asarray(model.predict(sample), dtype=float)
    finite_positive = bool(np.all(np.isfinite(predictions)) and np.all(predictions > 0))
    low, high = t.plausible_range
    out_of_range = float(np.mean((predictions < low) | (predictions > high)))

    bigger = sample.copy()
    bigger["house_size"] = pd.to_numeric(bigger["house_size"]) * (1 + t.size_increase)
    larger = np.asarray(model.predict(bigger), dtype=float)
    decreases = int(np.sum(larger < predictions * (1 - 1e-9)))

    unseen = sample.head(1).copy()
    for name in ("status", "state", "city", "zip_code", "brokered_by"):
        if name in unseen.columns:
            unseen[name] = UNSEEN
    try:
        unseen_prediction = float(np.asarray(model.predict(unseen), dtype=float)[0])
        unseen_ok = bool(np.isfinite(unseen_prediction) and unseen_prediction > 0)
    except Exception as error:
        unseen_prediction, unseen_ok = None, False
        unseen_error = str(error)
    else:
        unseen_error = None

    checks = {
        "finite_positive": finite_positive,
        "plausible_range": out_of_range <= t.max_out_of_range_share,
        "monotonic_house_size": decreases == 0,
        "unseen_categories": unseen_ok,
    }
    message = (
        f"G5 comportamiento: {percent(out_of_range, 2)} fuera de rango plausible, "
        f"{decreases} predicciones bajan al aumentar house_size "
        f"{percent(t.size_increase, 0)}, categorías desconocidas "
        + ("sin error" if unseen_ok else "con error")
    )
    detail = {
        "checks": checks,
        "rows": len(sample),
        "out_of_range_share": out_of_range,
        "monotonic_decreases": decreases,
        "unseen_prediction": unseen_prediction,
        "unseen_error": unseen_error,
    }
    return GateResult("G5_behavior", all(checks.values()), message, detail)


SERVING_THREADS = 1


def measure_latency_p95(model, X: pd.DataFrame, calls: int = 200, seed: int = 42) -> float:
    rows = X.sample(min(len(X), calls), random_state=seed, replace=len(X) < calls)
    timings = []
    with threadpool_limits(limits=SERVING_THREADS):
        model.predict(rows.iloc[[0]])
        for i in range(len(rows)):
            row = rows.iloc[[i]]
            start = time.perf_counter()
            model.predict(row)
            timings.append((time.perf_counter() - start) * 1000)
    return float(np.percentile(timings, 95))


def serialized_size_mb(model) -> float:
    return len(pickle.dumps(model)) / 1e6


def gate_budget(latency_p95_ms: float, size_mb: float, t: PromotionThresholds) -> GateResult:
    checks = {
        "latency": latency_p95_ms <= t.max_latency_p95_ms,
        "size": size_mb <= t.max_model_size_mb,
    }
    message = (
        f"G6 presupuesto: p95 de una predicción {decimal(latency_p95_ms, 1)} ms "
        f"(máximo {decimal(t.max_latency_p95_ms, 0)}), modelo de {decimal(size_mb, 1)} MB "
        f"(máximo {decimal(t.max_model_size_mb, 0)})"
    )
    detail = {"checks": checks, "latency_p95_ms": latency_p95_ms, "size_mb": size_mb}
    return GateResult("G6_budget", all(checks.values()), message, detail)


def compare_models(
    candidate,
    champion,
    data: EvaluationSet,
    thresholds: PromotionThresholds | None = None,
    latency_p95_ms: float | None = None,
    size_mb: float | None = None,
) -> PromotionDecision:
    t = thresholds or PromotionThresholds()
    y = np.asarray(data.y, dtype=float)
    pred_candidate = np.asarray(candidate.predict(data.X), dtype=float)
    candidate_metrics = regression_metrics(y, pred_candidate)
    if latency_p95_ms is None:
        latency_p95_ms = measure_latency_p95(candidate, data.X, t.latency_calls)
    if size_mb is None:
        size_mb = serialized_size_mb(candidate)
    behavior = gate_behavior(candidate, data.X, t)
    budget = gate_budget(latency_p95_ms, size_mb, t)

    if champion is None:
        gates = [behavior, budget]
        promote = behavior.passed and budget.passed
        reason = (
            "Se promueve: no había modelo productivo y el candidato pasa las pruebas de "
            "comportamiento y el presupuesto de servicio."
            if promote
            else "Se rechaza: " + "; ".join(g.message for g in gates if not g.passed) + "."
        )
        diagnostics = segment_table(y, pred_candidate, None, data.diagnostic_segments, t)
        return PromotionDecision(promote, reason, gates, candidate_metrics, None, diagnostics)

    pred_champion = np.asarray(champion.predict(data.X), dtype=float)
    champion_metrics = regression_metrics(y, pred_champion)
    diagnostics = segment_table(y, pred_candidate, pred_champion, data.diagnostic_segments, t)
    recent = np.asarray(data.recent, dtype=bool)
    gates = [
        gate_holdout(candidate_metrics, champion_metrics, t),
        gate_significance(ape(y, pred_candidate), ape(y, pred_champion), t),
        gate_recent(y, pred_candidate, pred_champion, recent, t),
        gate_segments(y, pred_candidate, pred_champion, data.segments, t),
        behavior,
        budget,
    ]
    promote = all(g.passed for g in gates)
    if promote:
        holdout = gates[0].detail
        reason = (
            f"Se promueve: MAPE {percent(candidate_metrics['mape'], 2)} frente a "
            f"{percent(champion_metrics['mape'], 2)} ({signed_percent(holdout['mape_change'])}), "
            f"MAE {signed_percent(holdout['mae_change'])}, R² {decimal(holdout['r2_change'], 3)}; "
            "la mejora es significativa y pasa segmentos, comportamiento y presupuesto."
        )
    else:
        reason = "Se rechaza: " + "; ".join(g.message for g in gates if not g.passed) + "."
    return PromotionDecision(
        promote, reason, gates, candidate_metrics, champion_metrics, diagnostics
    )
