import requests

from housing_ui.config import SETTINGS

ALIASES = ("champion", "challenger", "previous_champion")


def _get(path: str, params: dict) -> dict:
    response = requests.get(
        f"{SETTINGS.mlflow_url}/api/2.0/mlflow/{path}", params=params, timeout=SETTINGS.timeout
    )
    response.raise_for_status()
    return response.json()


def aliases() -> dict[str, int]:
    model = _get("registered-models/get", {"name": SETTINGS.model_name})["registered_model"]
    return {item["alias"]: int(item["version"]) for item in model.get("aliases", [])}


def version(number: int) -> dict:
    data = _get("model-versions/get", {"name": SETTINGS.model_name, "version": str(number)})
    mv = data["model_version"]
    mv["tags"] = {tag["key"]: tag["value"] for tag in mv.get("tags", [])}
    return mv


def version_url(number: int) -> str:
    return f"{SETTINGS.mlflow_public_url}/#/models/{SETTINGS.model_name}/versions/{number}"
