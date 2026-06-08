"""Testes da API"""
import pytest
from fastapi.testclient import TestClient

from src import api


@pytest.fixture
def client():
    with TestClient(api.app) as c:
        yield c


def test_health_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "model_loaded" in body


def test_root_redirects_to_docs(client):
    resp = client.get("/", follow_redirects=False)
    assert resp.status_code in (307, 308)
    assert resp.headers["location"].endswith("/docs")


def test_predict_returns_valid_shape(client, sample_payload):
    resp = client.post("/predict", json=sample_payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["prediction"] in (0, 1)
    assert 0.0 <= body["probability"] <= 1.0
    assert body["label"] in ("Churn", "Stay")


def test_predict_rejects_invalid_senior_citizen(client, sample_payload):
    bad = {**sample_payload, "SeniorCitizen": "No"}
    resp = client.post("/predict", json=bad)
    assert resp.status_code == 422


def test_predict_rejects_missing_field(client, sample_payload):
    bad = {k: v for k, v in sample_payload.items() if k != "tenure"}
    resp = client.post("/predict", json=bad)
    assert resp.status_code == 422
