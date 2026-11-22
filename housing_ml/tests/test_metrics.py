import numpy as np
import pytest

from housing_ml.metrics import paired_bootstrap, price_band, regression_metrics


def test_regression_metrics():
    metrics = regression_metrics([100, 200, 400], [110, 180, 400])
    assert metrics["mape"] == pytest.approx((0.1 + 0.1 + 0) / 3)
    assert metrics["mdape"] == pytest.approx(0.1)
    assert metrics["mae"] == pytest.approx(10)
    assert metrics["n"] == 3


def test_price_bands():
    assert price_band([100_000, 300_000, 499_999, 2_000_000]).tolist() == [
        "<300k",
        "300k-500k",
        "300k-500k",
        ">500k",
    ]


def test_bootstrap_detects_real_improvement():
    rng = np.random.default_rng(0)
    diff = rng.normal(-0.02, 0.05, 20_000)
    ci = paired_bootstrap(diff, samples=300)
    assert ci["high"] < 0


def test_bootstrap_includes_zero_without_difference():
    rng = np.random.default_rng(0)
    diff = rng.normal(0, 0.05, 20_000)
    ci = paired_bootstrap(diff, samples=300)
    assert ci["low"] < 0 < ci["high"]


def test_bootstrap_subsamples_large_inputs():
    ci = paired_bootstrap(np.zeros(200_000), samples=10, max_rows=1_000)
    assert ci["rows"] == 1_000


def test_size_bands():
    from housing_ml.metrics import size_band

    assert size_band([800, 1_500, 2_999, 5_000, float("nan")]).tolist() == [
        "<1500",
        "1500-3000",
        "1500-3000",
        ">3000",
        "unknown",
    ]
