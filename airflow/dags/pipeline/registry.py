import os

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import sklearn
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from mlflow.models import infer_signature

from pipeline.config import (
    CHALLENGER,
    CHAMPION,
    MLFLOW_EXPERIMENT,
    MLFLOW_TRACKING_URI,
    MODEL_NAME,
    PREVIOUS_CHAMPION,
)

SKOPS_TRUSTED_TYPES = [
    "housing_ml.features.FeatureBuilder",
    "housing_ml.model.MappedTargetEncoder",
    "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor",
]


def configure() -> MlflowClient:
    os.environ.setdefault("MLFLOW_ENABLE_PROXY_MULTIPART_DOWNLOAD", "false")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    return MlflowClient()


def pip_requirements() -> list[str]:
    import scipy
    import skops

    import housing_ml

    return [
        f"scikit-learn=={sklearn.__version__}",
        f"numpy=={np.__version__}",
        f"pandas=={pd.__version__}",
        f"scipy=={scipy.__version__}",
        f"skops=={skops.__version__}",
        f"housing-ml=={housing_ml.__version__}",
    ]


def version_by_alias(alias: str) -> dict | None:
    client = configure()
    try:
        mv = client.get_model_version_by_alias(MODEL_NAME, alias)
    except MlflowException:
        return None
    reference = mv.tags.get("holdout_mape")
    return {
        "version": int(mv.version),
        "run_id": mv.run_id,
        "reference_mape": float(reference) if reference else None,
    }


def champion() -> dict | None:
    return version_by_alias(CHAMPION)


def load_version(version: int):
    configure()
    return mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{version}")


def load_uri(model_uri: str):
    configure()
    return mlflow.sklearn.load_model(model_uri)


def log_model(model, sample: pd.DataFrame):
    signature = infer_signature(sample, model.predict(sample))
    return mlflow.sklearn.log_model(
        model,
        name="model",
        signature=signature,
        pip_requirements=pip_requirements(),
        skops_trusted_types=SKOPS_TRUSTED_TYPES,
    )


def register(model_uri: str, tags: dict) -> int:
    client = configure()
    mv = mlflow.register_model(model_uri, MODEL_NAME)
    for key, value in tags.items():
        client.set_model_version_tag(MODEL_NAME, mv.version, key, str(value))
    client.set_registered_model_alias(MODEL_NAME, CHALLENGER, mv.version)
    return int(mv.version)


def tag_version(version: int, tags: dict) -> None:
    client = configure()
    for key, value in tags.items():
        client.set_model_version_tag(MODEL_NAME, str(version), key, str(value))


def promote(version: int) -> int | None:
    client = configure()
    current = champion()
    if current and current["version"] != version:
        client.set_registered_model_alias(MODEL_NAME, PREVIOUS_CHAMPION, str(current["version"]))
    client.set_registered_model_alias(MODEL_NAME, CHAMPION, str(version))
    return current["version"] if current else None
