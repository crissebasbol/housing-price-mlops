import numpy as np
import pandas as pd
import pytest

from housing_ml.features import MODEL_INPUT_COLUMNS
from housing_ml.metrics import regression_metrics
from housing_ml.model import build_model, describe_model, iterations


@pytest.fixture(scope="module")
def trained():
    from conftest import make_records

    frame = pd.DataFrame(make_records(3_000, seed=1))
    model = build_model({"max_iter": 100}).fit(frame[list(MODEL_INPUT_COLUMNS)], frame["price"])
    return model, frame


def test_model_learns_prices(trained):
    model, frame = trained
    metrics = regression_metrics(frame["price"], model.predict(frame[list(MODEL_INPUT_COLUMNS)]))
    assert metrics["mape"] < 0.2
    assert iterations(model) > 0


def test_predictions_are_positive_and_monotonic_in_size(trained):
    model, frame = trained
    sample = frame[list(MODEL_INPUT_COLUMNS)].head(200)
    base = model.predict(sample)
    bigger = model.predict(sample.assign(house_size=sample["house_size"] * 1.3))
    assert np.all(base > 0)
    assert np.all(bigger >= base * (1 - 1e-9))


def test_model_accepts_raw_api_fields_and_unknown_categories(trained):
    model, _ = trained
    request = pd.DataFrame(
        [
            {
                "status": "ready_to_build",
                "bed": 3,
                "bath": 2,
                "acre_lot": 0.2,
                "house_size": 1_800,
                "prev_sold_date": None,
                "city": "Boston",
                "state": "Massachusetts",
                "zip_code": "02134",
                "brokered_by": None,
            }
        ]
    )
    prediction = model.predict(request)
    assert np.isfinite(prediction).all() and prediction[0] > 0


def test_describe_model_is_flat():
    description = describe_model()
    assert description["estimator"] == "HistGradientBoostingRegressor"
    assert all(not isinstance(v, dict) for v in description.values())


def test_mapped_encoder_matches_sklearn_at_inference():
    from sklearn.preprocessing import TargetEncoder

    from housing_ml.model import MappedTargetEncoder

    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        {"zip": rng.choice(["a", "b", "c", "d"], 500), "state": rng.choice(["x", "y"], 500)}
    )
    y = rng.normal(10, 1, 500)
    mapped = MappedTargetEncoder().fit(X, y)
    reference = TargetEncoder(target_type="continuous", random_state=42).fit(X, y)
    probe = pd.DataFrame({"zip": ["a", "d", "zzz"], "state": ["y", "x", "x"]})
    np.testing.assert_allclose(mapped.transform(probe).to_numpy(), reference.transform(probe))
