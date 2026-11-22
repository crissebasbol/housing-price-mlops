import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import TargetEncoder

from housing_ml.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, FeatureBuilder

MODEL_PARAMS = {
    "max_iter": 1500,
    "learning_rate": 0.1,
    "max_leaf_nodes": 127,
    "min_samples_leaf": 50,
    "l2_regularization": 1.0,
    "early_stopping": True,
    "validation_fraction": 0.1,
    "n_iter_no_change": 20,
    "random_state": 42,
}
MONOTONIC_CONSTRAINTS = {"house_size": 1, "bath": 1}
TARGET_TRANSFORM = "log1p"


class MappedTargetEncoder(TransformerMixin, BaseEstimator):
    """TargetEncoder de scikit-learn al entrenar (con validación cruzada) y diccionarios al servir.

    TargetEncoder.transform revisa todas las categorías conocidas en cada llamada (decenas de
    milisegundos por fila con ~100 mil categorías); un diccionario responde en microsegundos.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def fit(self, X, y):
        self.fit_transform(X, y)
        return self

    def fit_transform(self, X, y=None, **fit_params) -> pd.DataFrame:
        frame = pd.DataFrame(X)
        encoder = TargetEncoder(target_type="continuous", random_state=self.random_state)
        encoded = encoder.fit_transform(frame, y)
        self.feature_names_in_ = np.asarray(frame.columns, dtype=object)
        self.default_ = float(encoder.target_mean_)
        self.mappings_ = {
            name: dict(zip(categories.tolist(), values.tolist(), strict=True))
            for name, categories, values in zip(
                frame.columns, encoder.categories_, encoder.encodings_, strict=True
            )
        }
        return pd.DataFrame(encoded, columns=frame.columns, index=frame.index)

    def transform(self, X) -> pd.DataFrame:
        frame = pd.DataFrame(X)
        return pd.DataFrame(
            {
                name: frame[name].map(self.mappings_[name]).astype(float).fillna(self.default_)
                for name in self.feature_names_in_
            },
            index=frame.index,
        )

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.asarray(self.feature_names_in_, dtype=object)


def build_model(params: dict | None = None) -> TransformedTargetRegressor:
    encoder = ColumnTransformer(
        [
            ("numeric", "passthrough", list(NUMERIC_FEATURES)),
            ("categorical", MappedTargetEncoder(), list(CATEGORICAL_FEATURES)),
        ],
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")
    regressor = HistGradientBoostingRegressor(
        monotonic_cst=MONOTONIC_CONSTRAINTS, **{**MODEL_PARAMS, **(params or {})}
    )
    pipeline = Pipeline([("features", FeatureBuilder()), ("encode", encoder), ("model", regressor)])
    return TransformedTargetRegressor(
        regressor=pipeline, func=np.log1p, inverse_func=np.expm1, check_inverse=False
    )


def describe_model(params: dict | None = None) -> dict:
    return {
        "estimator": "HistGradientBoostingRegressor",
        "target_transform": TARGET_TRANSFORM,
        "numeric_features": ",".join(NUMERIC_FEATURES),
        "categorical_features": ",".join(CATEGORICAL_FEATURES),
        "categorical_encoder": "TargetEncoder (cross-fitting) + dictionary lookup",
        "monotonic_constraints": ",".join(f"{k}:+" for k in MONOTONIC_CONSTRAINTS),
        **{f"model_{k}": v for k, v in {**MODEL_PARAMS, **(params or {})}.items()},
    }


def iterations(model: TransformedTargetRegressor) -> int:
    return int(model.regressor_.named_steps["model"].n_iter_)
