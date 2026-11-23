from dataclasses import dataclass, field

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

END_OF_DATA_MESSAGE = "Ya se recolectó toda la información minima necesaria"


class DataApiError(RuntimeError):
    pass


@dataclass
class DataApiResponse:
    outcome: str
    status_code: int
    batch_number: int | None = None
    records: list[dict] = field(default_factory=list)


def _session() -> requests.Session:
    # Cada GET a /data consume un lote: solo se reintenta si la petición no llegó a la API
    # (conexión rechazada o gateway caído), nunca después de enviarla.
    retry = Retry(
        total=3,
        connect=3,
        read=0,
        status=3,
        status_forcelist=(502, 503, 504),
        backoff_factor=2,
        allowed_methods=("GET",),
        raise_on_status=False,
    )
    session = requests.Session()
    session.mount("http://", HTTPAdapter(max_retries=retry))
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def _detail(response: requests.Response) -> str:
    try:
        return str(response.json().get("detail", ""))
    except ValueError:
        return response.text[:200]


class DataApiClient:
    def __init__(
        self,
        base_url: str,
        group: int,
        day: str,
        timeout: float = 180,
        session: requests.Session | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.params = {"group_number": group, "day": day}
        self.timeout = (5, timeout)
        self.session = session or _session()

    def _get(self, path: str, params: dict | None = None) -> requests.Response:
        try:
            return self.session.get(f"{self.base_url}{path}", params=params, timeout=self.timeout)
        except requests.RequestException as error:
            raise DataApiError(f"No se pudo consultar {path}: {error}") from error

    def health(self) -> None:
        response = self._get("/health")
        if response.status_code != 200:
            raise DataApiError(f"/health respondió {response.status_code}")

    def fetch(self) -> DataApiResponse:
        response = self._get("/data", self.params)
        if response.status_code == 400:
            detail = _detail(response)
            if detail == END_OF_DATA_MESSAGE:
                return DataApiResponse("end_of_data", 400)
            raise DataApiError(f"/data respondió 400: {detail}")
        if response.status_code != 200:
            raise DataApiError(f"/data respondió {response.status_code}: {_detail(response)}")
        try:
            body = response.json()
        except ValueError as error:
            raise DataApiError("/data no devolvió JSON") from error
        if not isinstance(body, dict) or not isinstance(body.get("data"), list):
            raise DataApiError("/data devolvió una respuesta sin la lista 'data'")
        records = body["data"]
        batch_number = body.get("batch_number")
        return DataApiResponse(
            "data" if records else "empty",
            200,
            int(batch_number) if batch_number is not None else None,
            records,
        )

    def restart(self) -> dict:
        response = self._get("/restart_data_generation", self.params)
        if response.status_code != 200:
            raise DataApiError(f"El reinicio respondió {response.status_code}: {_detail(response)}")
        return response.json()
