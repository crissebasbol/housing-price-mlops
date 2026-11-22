import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

PRICE_BANDS = ((300_000, "<300k"), (500_000, "300k-500k"), (np.inf, ">500k"))
SIZE_BANDS = ((1_500, "<1500"), (3_000, "1500-3000"), (np.inf, ">3000"))


def ape(y_true, y_pred) -> np.ndarray:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return np.abs(y_pred - y_true) / np.abs(y_true)


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    errors = ape(y_true, y_pred)
    return {
        "mape": float(errors.mean()),
        "mdape": float(np.median(errors)),
        "mae": float(np.mean(np.abs(y_pred - y_true))),
        "rmse": float(np.sqrt(np.mean((y_pred - y_true) ** 2))),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan"),
        "n": int(len(y_true)),
    }


def _bands(values, bands) -> pd.Series:
    values = pd.Series(np.asarray(values, dtype=float))
    labels = pd.Series("unknown", index=values.index, dtype=object)
    lower = -np.inf
    for upper, label in bands:
        labels[(values >= lower) & (values < upper)] = label
        lower = upper
    return labels


def price_band(prices) -> pd.Series:
    return _bands(prices, PRICE_BANDS)


def size_band(house_sizes) -> pd.Series:
    return _bands(house_sizes, SIZE_BANDS)


def paired_bootstrap(
    differences,
    samples: int = 1_000,
    max_rows: int = 50_000,
    seed: int = 42,
    alpha: float = 0.05,
    chunk: int = 100,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    diff = np.asarray(differences, dtype=float)
    if len(diff) > max_rows:
        diff = rng.choice(diff, max_rows, replace=False)
    means = []
    for start in range(0, samples, chunk):
        size = min(chunk, samples - start)
        idx = rng.integers(0, len(diff), size=(size, len(diff)))
        means.append(diff[idx].mean(axis=1))
    means = np.concatenate(means)
    return {
        "mean": float(diff.mean()),
        "low": float(np.quantile(means, alpha / 2)),
        "high": float(np.quantile(means, 1 - alpha / 2)),
        "rows": int(len(diff)),
        "samples": int(samples),
    }
