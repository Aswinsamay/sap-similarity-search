import pytest
from fastapi.testclient import TestClient

import app as api
from similarity.preprocessing import PROCESSED_PATH
from similarity.search import find_similar_products

pytestmark = pytest.mark.skipif(not PROCESSED_PATH.exists(), reason="run `python -m similarity.preprocessing` first")

QUERY = "26d41bdc1495de290bc8e6062d927729"
URL = "/find_similar_products"


@pytest.fixture(scope="module")
def client():
    with TestClient(api.app) as client:   # `with` runs the startup step, like uvicorn does
        yield client


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "products": 30_000}


def test_same_answer_as_the_function(client):
    response = client.get(URL, params={"product_id": QUERY, "num_similar": 5})
    assert response.status_code == 200
    assert response.json() == find_similar_products(QUERY, 5)


def test_num_similar_defaults_to_10(client):
    assert len(client.get(URL, params={"product_id": QUERY}).json()) == 10


def test_up_to_the_maximum(client):
    assert len(client.get(URL, params={"product_id": QUERY, "num_similar": 100}).json()) == 100


def test_shorter_answer_is_the_start_of_a_longer_one(client):
    three = client.get(URL, params={"product_id": QUERY, "num_similar": 3}).json()
    ten = client.get(URL, params={"product_id": QUERY, "num_similar": 10}).json()
    assert three == ten[:3]


def test_unknown_product_is_404(client):
    response = client.get(URL, params={"product_id": "not-a-product"})
    assert response.status_code == 404
    assert response.json() == {"detail": "product_id not found: not-a-product"}


@pytest.mark.parametrize(
    "params",
    [
        {},                                            # product_id missing
        {"product_id": ""},                            # product_id empty
        {"product_id": QUERY, "num_similar": 0},       # below 1
        {"product_id": QUERY, "num_similar": 101},     # above the maximum
        {"product_id": QUERY, "num_similar": "ten"},   # not a number
    ],
)
def test_bad_request_is_422(client, params):
    assert client.get(URL, params=params).status_code == 422


def test_unexpected_error_is_500(client, monkeypatch):
    def broken(product_id):
        raise RuntimeError("something internal")

    monkeypatch.setattr(api, "top_similar", broken)
    response = client.get(URL, params={"product_id": QUERY})
    assert response.status_code == 500
    assert response.json() == {"detail": "internal error while searching"}   # internals not shown


def test_repeat_request_is_served_from_the_cache(client):
    api.top_similar.cache_clear()
    client.get(URL, params={"product_id": QUERY, "num_similar": 3})
    client.get(URL, params={"product_id": QUERY, "num_similar": 7})
    client.get(URL, params={"product_id": "not-a-product"})
    info = api.top_similar.cache_info()
    assert (info.misses, info.hits, info.currsize) == (1, 1, 1)   # one search; unknown id not stored
