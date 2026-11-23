import requests

from housing_ui.config import SETTINGS

HEADERS = {"X-Traffic-Source": "streamlit"}


class ApiError(Exception):
    pass


def _detail(response: requests.Response) -> str:
    try:
        detail = response.json().get("detail", response.text)
    except ValueError:
        return response.text[:300]
    if isinstance(detail, list):
        return "; ".join(
            f"{'.'.join(str(p) for p in item.get('loc', [])[1:])}: {item.get('msg', '')}"
            for item in detail
        )
    return str(detail)


def _call(method: str, path: str, **kwargs) -> dict:
    try:
        response = requests.request(
            method, f"{SETTINGS.api_url}{path}", headers=HEADERS, timeout=SETTINGS.timeout, **kwargs
        )
    except requests.RequestException as error:
        raise ApiError(f"La API no respondió: {error}") from error
    if response.status_code >= 400:
        raise ApiError(f"La API respondió {response.status_code}: {_detail(response)}")
    return response.json()


def predict(payload: dict) -> dict:
    return _call("POST", "/predict", json=payload)


def explain(payload: dict) -> dict:
    return _call("POST", "/explain", json=payload)


def feedback(request_id: str, actual_price: float) -> dict:
    return _call("POST", "/feedback", json={"request_id": request_id, "actual_price": actual_price})


def model_info() -> dict:
    return _call("GET", "/model-info")
