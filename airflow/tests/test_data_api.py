import pytest
import requests

from pipeline.data_api import END_OF_DATA_MESSAGE, DataApiClient, DataApiError, _session


class FakeResponse:
    def __init__(self, status_code: int, body=None, text: str = ""):
        self.status_code = status_code
        self._body = body
        self.text = text

    def json(self):
        if self._body is None:
            raise ValueError("no es JSON")
        return self._body


class FakeSession:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, params, timeout))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def make_client(*responses) -> tuple[DataApiClient, FakeSession]:
    session = FakeSession(*responses)
    return DataApiClient("http://data-api/", 1, "Tuesday", timeout=30, session=session), session


def test_batch_with_rows():
    body = {"group_number": 1, "batch_number": 2, "day": "Tuesday", "data": [{"price": 1.0}]}
    client, session = make_client(FakeResponse(200, body))
    response = client.fetch()
    assert response.outcome == "data"
    assert response.batch_number == 2
    assert response.records == [{"price": 1.0}]
    url, params, timeout = session.calls[0]
    assert url == "http://data-api/data"
    assert params == {"group_number": 1, "day": "Tuesday"}
    assert timeout == (5, 30)


def test_empty_batch():
    client, _ = make_client(FakeResponse(200, {"batch_number": 3, "data": []}))
    assert client.fetch().outcome == "empty"


def test_end_of_data_is_not_an_error():
    client, _ = make_client(FakeResponse(400, {"detail": END_OF_DATA_MESSAGE}))
    response = client.fetch()
    assert response.outcome == "end_of_data"
    assert response.records == []


@pytest.mark.parametrize(
    "response",
    [
        FakeResponse(400, {"detail": "Número de grupo inválido"}),
        FakeResponse(500, None, "Internal Server Error"),
        FakeResponse(200, None, "<html>"),
        FakeResponse(200, {"batch_number": 1}),
        FakeResponse(200, {"data": "no es lista"}),
    ],
)
def test_unexpected_responses_raise(response):
    client, _ = make_client(response)
    with pytest.raises(DataApiError):
        client.fetch()


def test_connection_errors_raise():
    client, _ = make_client(requests.ConnectionError("rechazada"))
    with pytest.raises(DataApiError, match="No se pudo consultar"):
        client.fetch()


def test_restart_and_health():
    client, session = make_client(
        FakeResponse(200, {"ok": True, "status": "reset ok"}), FakeResponse(200, {"status": "OK"})
    )
    assert client.restart()["status"] == "reset ok"
    client.health()
    assert session.calls[0][0].endswith("/restart_data_generation")


def test_reads_are_never_retried():
    retry = _session().get_adapter("http://data-api").max_retries
    assert retry.read == 0
    assert set(retry.status_forcelist) == {502, 503, 504}
